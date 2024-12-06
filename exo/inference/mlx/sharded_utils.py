# Adapted from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py

import glob
import importlib
import json
import logging
import asyncio
import aiohttp
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, List, Callable
from PIL import Image
from io import BytesIO
import base64
import traceback

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoProcessor

from mlx_lm.tokenizer_utils import load_tokenizer, TokenizerWrapper

from exo import DEBUG
from exo.inference.tokenizers import resolve_tokenizer
from ..shard import Shard


class ModelNotFoundError(Exception):
  def __init__(self, message):
    self.message = message
    super().__init__(self.message)


# 模型类型的重映射关系，用于处理兼容性
MODEL_REMAPPING = {
  "mistral": "llama",  # mistral 模型与 llama 架构兼容
  "phi-msft": "phixtral",
}


def _get_classes(config: dict):
  """
  根据配置获取模型和模型参数类。
  
  Args:
    config (dict): 模型配置字典
  
  Returns:
    tuple: 返回 (Model类, ModelArgs类) 的元组
  """
  model_type = config["model_type"]
  model_type = MODEL_REMAPPING.get(model_type, model_type)
  try:
    arch = importlib.import_module(f"exo.inference.mlx.models.{model_type}")
  except ImportError:
    msg = f"Model type {model_type} not supported."
    logging.error(msg)
    traceback.print_exc()
    raise ValueError(msg)

  return arch.Model, arch.ModelArgs


def load_config(model_path: Path) -> dict:
  try:
    with open(model_path/"config.json", "r") as f:
      config = json.load(f)
  except FileNotFoundError:
    logging.error(f"Config file not found in {model_path}")
    raise
  return config

def load_model_shard(
  model_path: Path,
  shard: Shard,
  lazy: bool = False,
  model_config: dict = {},
) -> nn.Module:
  """
  从指定路径加载并初始化模型分片。
  
  Args:
    model_path (Path): 模型路径
    shard (Shard): 分片配置对象，定义了当前分片的层范围
    lazy (bool): 如果为False，立即加载模型参数到内存；如果为True，则按需加载
    model_config (dict): 额外的模型配置参数
  
  Returns:
    nn.Module: 加载并初始化好的模型
  """
  config = load_config(model_path)
  config.update(model_config)

  # 添加分片配置信息到模型配置中
  config["shard"] = {
    "model_id": model_path.name,
    "start_layer": shard.start_layer,
    "end_layer": shard.end_layer,
    "n_layers": shard.n_layers,
  }

  weight_files = glob.glob(str(model_path/"model*.safetensors"))

  if not weight_files:
    # Try weight for back-compat
    weight_files = glob.glob(str(model_path/"weight*.safetensors"))

  if not weight_files:
    logging.error(f"No safetensors found in {model_path}")
    raise FileNotFoundError(f"No safetensors found in {model_path}")

  weights = {}
  for wf in sorted(weight_files):
    if DEBUG >= 8:
      layer_nums = set()
      for k in mx.load(wf):
        if k.startswith("model.layers."):
          layer_num = int(k.split(".")[2])
          layer_nums.add(layer_num)
        if k.startswith("language_model.model.layers."):
          layer_num = int(k.split(".")[3])
          layer_nums.add(layer_num)
      print(f"\"{wf.split('/')[-1]}\": {sorted(layer_nums)},")

    weights.update(mx.load(wf))

  model_class, model_args_class = _get_classes(config=config)

  class ShardedModel(model_class):
    """
    对模型类进行包装，添加分片支持
    """
    def __init__(self, args):
      super().__init__(args)
      self.shard = Shard(args.shard.model_id, args.shard.start_layer, args.shard.end_layer, args.shard.n_layers)

    def __call__(self, x, *args, **kwargs):
      y = super().__call__(x, *args, **kwargs)
      return y

  model_args = model_args_class.from_dict(config)
  model = ShardedModel(model_args)

  if hasattr(model, "sanitize"):
    weights = model.sanitize(weights)

  if (quantization := config.get("quantization", None)) is not None:
    # Handle legacy models which may not have everything quantized
    def class_predicate(p, m):
      if not hasattr(m, "to_quantized"):
        return False
      return f"{p}.scales" in weights

    nn.quantize(
      model,
      **quantization,
      class_predicate=class_predicate,
    )

  model.load_weights(list(weights.items()), strict=True)

  if not lazy:
    mx.eval(model.parameters())

  model.eval()
  return model

async def load_shard(
  model_path: str,
  shard: Shard,
  tokenizer_config={},
  model_config={},
  adapter_path: Optional[str] = None,
  lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
  """
  异步加载模型分片和对应的分词器
  
  Args:
    model_path: 模型路径
    shard: 分片配置
    tokenizer_config: 分词器配置
    model_config: 模型配置
    adapter_path: 适配器路径（可选）
    lazy: 是否延迟加载模型参数
    
  Returns:
    tuple: 返回 (模型, 分词器) 的元组
  """
  model = load_model_shard(model_path, shard, lazy, model_config)

  # TODO: figure out a generic solution
  if model.model_type == "llava":
    processor = AutoProcessor.from_pretrained(model_path)
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.encode = processor.tokenizer.encode
    return model, processor
  else:
    tokenizer = await resolve_tokenizer(model_path)
    return model, tokenizer


async def get_image_from_str(_image_str: str):
  """
  从URL或base64字符串加载图像
  
  Args:
    _image_str: 图像URL或base64编码的图像数据
    
  Returns:
    PIL.Image: 加载的RGB格式图像
    
  Raises:
    ValueError: 当输入格式不正确时抛出
  """
  image_str = _image_str.strip()

  if image_str.startswith("http"):
    async with aiohttp.ClientSession() as session:
      async with session.get(image_str, timeout=10) as response:
        content = await response.read()
        return Image.open(BytesIO(content)).convert("RGB")
  elif image_str.startswith("data:image/"):
    # Extract the image format and base64 data
    format_prefix, base64_data = image_str.split(";base64,")
    image_format = format_prefix.split("/")[1].lower()
    if DEBUG >= 2: print(f"{image_str=} {image_format=}")
    imgdata = base64.b64decode(base64_data)
    img = Image.open(BytesIO(imgdata))

    # Convert to RGB if not already
    if img.mode != "RGB":
      img = img.convert("RGB")

    return img
  else:
    raise ValueError("Invalid image_str format. Must be a URL or a base64 encoded image.")
