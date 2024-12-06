import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import top_p_sampling
from ..inference_engine import InferenceEngine
from .stateful_model import StatefulModel
from .sharded_utils import load_shard
from ..shard import Shard
from typing import Dict, Optional, Tuple
from exo.download.shard_download import ShardDownloader
import asyncio
from concurrent.futures import ThreadPoolExecutor

def sample_logits(
  logits: mx.array,
  temp: float = 0.0,
  top_p: float = 1.0,
  logit_bias: Optional[Dict[int, float]] = None
) -> Tuple[mx.array, float]:
  """
  对模型输出的logits进行采样,生成下一个token
  Args:
      logits: 模型输出的logits分布
      temp: 温度参数,控制采样随机性,0表示贪婪采样
      top_p: nucleus采样的概率阈值
      logit_bias: token的偏置字典
  Returns:
      采样得到的token
  """
  if logit_bias:
    indices = mx.array(list(logit_bias.keys()))
    values = mx.array(list(logit_bias.values()))
    logits[:, indices] += values

  if temp == 0:
    token = mx.argmax(logits, axis=-1)
  else:
    if top_p > 0 and top_p < 1.0:
      token = top_p_sampling(logits, top_p, temp)
    else:
      token = mx.random.categorical(logits*(1/temp))

  return token

class MLXDynamicShardInferenceEngine(InferenceEngine):
  """MLX推理引擎的动态分片实现"""
  
  def __init__(self, shard_downloader: ShardDownloader):
    """
    初始化推理引擎
    Args:
        shard_downloader: 用于下载模型分片的下载器
    """
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  async def sample(self, x, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
    """
    对模型输出进行采样
    Args:
        x: 输入张量
        temp: 温度参数
        top_p: nucleus采样参数
    Returns:
        采样得到的token
    """
    y = mx.array(x)
    logits = y[:, -1, :]
    out = np.array(sample_logits(logits, temp=temp, top_p=top_p), dtype=int)
    return out

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    """
    将文本编码为token序列
    Args:
        shard: 模型分片
        prompt: 输入文本
    Returns:
        编码后的token序列
    """
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    return np.array(tokens)

  async def decode(self, shard: Shard, tokens) -> str:
    """
    将token序列解码为文本
    Args:
        shard: 模型分片
        tokens: token序列
    Returns:
        解码后的文本
    """
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.decode, tokens)
    return tokens
    
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray) -> np.ndarray:
    """
    执行模型推理
    Args:
        request_id: 请求ID
        shard: 模型分片
        input_data: 输入数据
    Returns:
        模型输出
    """
    await self.ensure_shard(shard)
    output_data: np.ndarray = np.array(await asyncio.get_running_loop().run_in_executor(self.executor, self.model, mx.array(input_data), request_id))
    return output_data

  async def ensure_shard(self, shard: Shard):
    """
    确保指定的模型分片已加载
    Args:
        shard: 需要加载的模型分片
    """
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)

    if self.shard != shard:
      loop = asyncio.get_running_loop()

      def load_shard_wrapper():
        return asyncio.run(load_shard(model_path, shard))

      model_shard, self.tokenizer = await loop.run_in_executor(self.executor, load_shard_wrapper)
      self.shard = shard
      self.model = await loop.run_in_executor(self.executor, StatefulModel, model_shard) 
