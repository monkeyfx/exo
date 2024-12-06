import numpy as np
import os
from exo.helpers import DEBUG  # 确保导入DEBUG

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard


class InferenceEngine(ABC):
  """
  推理引擎的抽象基类，定义了推理引擎的基本接口。
  """
  @abstractmethod
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    """
    编码输入提示为模型可处理的张量。
    
    Args:
      shard (Shard): 模型分片信息
      prompt (str): 输入提示文本
    
    Returns:
      np.ndarray: 编码后的张量
    """
    pass
  
  @abstractmethod
  async def sample(self, x: np.ndarray) -> np.ndarray:
    """
    从模型输出中采样结果。
    
    Args:
      x (np.ndarray): 模型输出张量
    
    Returns:
      np.ndarray: 采样后的结果
    """
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    """
    解码模型输出的张量为可读文本。
    
    Args:
      shard (Shard): 模型分片信息
      tokens (np.ndarray): 模型输出的张量
    
    Returns:
      str: 解码后的文本
    """
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray) -> np.ndarray:
    """
    执行推理并返回结果张量。
    
    Args:
      request_id (str): 请求ID
      shard (Shard): 模型分片信息
      input_data (np.ndarray): 输入数据张量
    
    Returns:
      np.ndarray: 推理结果张量
    """
    pass
  
  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str) -> np.ndarray:
    """
    对输入提示进行推理并返回结果。
    
    Args:
      request_id (str): 请求ID
      shard (Shard): 模型分片信息
      prompt (str): 输入提示文本
    
    Returns:
      np.ndarray: 推理结果张量
    """
    tokens = await self.encode(shard, prompt)
    x = tokens.reshape(1, -1)
    output_data = await self.infer_tensor(request_id, shard, x)
    return output_data 

inference_engine_classes = {
  "mlx": "MLXDynamicShardInferenceEngine",
  "tinygrad": "TinygradDynamicShardInferenceEngine",
  "dummy": "DummyInferenceEngine",
}

def get_inference_engine(inference_engine_name: str, shard_downloader: 'ShardDownloader'):
  """
  根据名称获取对应的推理引擎实例。
  
  Args:
    inference_engine_name (str): 推理引擎名称
    shard_downloader (ShardDownloader): 分片下载器实例
  
  Returns:
    InferenceEngine: 对应的推理引擎实例
  
  Raises:
    ValueError: 如果不支持指定的推理引擎名称
  """
  if DEBUG >= 2:
    print(f"get_inference_engine called with: {inference_engine_name}")
  if inference_engine_name == "mlx":
    from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine

    return MLXDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "tinygrad":
    from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
    import tinygrad.helpers
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))

    return TinygradDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "dummy":
    from exo.inference.dummy_inference_engine import DummyInferenceEngine
    return DummyInferenceEngine()
  raise ValueError(f"Unsupported inference engine: {inference_engine_name}")
