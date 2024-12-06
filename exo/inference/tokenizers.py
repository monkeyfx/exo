import traceback
from aiofiles import os as aios
from os import PathLike
from pathlib import Path
from typing import Union
from transformers import AutoTokenizer, AutoProcessor
import numpy as np
from exo.download.hf.hf_helpers import get_local_snapshot_dir
from exo.helpers import DEBUG

'''
主要功能说明：
DummyTokenizer 类：
用于测试环境的虚拟分词器,提供基本的分词接口但返回固定值,主要用于开发和测试阶段
resolve_tokenizer 函数：
主要入口函数,处理模型分词器的解析逻辑,支持从本地或远程加载分词器,包含错误处理和日志记录
_resolve_tokenizer 函数：
内部实现函数,按优先级尝试不同的加载方式：
1. 首先尝试使用 AutoProcessor,如果失败则尝试使用 AutoTokenizer,都失败则抛出错误,
2. 为处理器添加必要的兼容性属性
'''
class DummyTokenizer:
    """
    虚拟分词器类，用于测试和开发
    提供基本的分词器接口实现，返回固定的测试值
    """
    def __init__(self):
        # 设置结束符token ID和词汇表大小
        self.eos_token_id = 69
        self.vocab_size = 1000

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        """应用聊天模板到消息列表"""
        return "dummy_tokenized_prompt"

    def encode(self, text):
        """将文本编码为token ID"""
        return np.array([1])

    def decode(self, tokens):
        """将token ID解码为文本"""
        return "dummy" * len(tokens)


async def resolve_tokenizer(model_id: str):
    """
    解析并获取指定模型的分词器
    
    Args:
        model_id: 模型标识符
    
    Returns:
        分词器实例，如果是dummy模型则返回DummyTokenizer
    """
    # 如果是dummy模型，返回虚拟分词器
    if model_id == "dummy":
        return DummyTokenizer()
    
    # 获取本地模型路径
    local_path = await get_local_snapshot_dir(model_id)
    if DEBUG >= 2: print(f"Checking if local path exists to load tokenizer from local {local_path=}")
    
    try:
        # 如果本地路径存在，尝试从本地加载
        if local_path and await aios.path.exists(local_path):
            if DEBUG >= 2: print(f"Resolving tokenizer for {model_id=} from {local_path=}")
            return await _resolve_tokenizer(local_path)
    except:
        # 如果本地加载失败，尝试从远程加载
        if DEBUG >= 5: print(f"Local check for {local_path=} failed. Resolving tokenizer for {model_id=} normally...")
        if DEBUG >= 5: traceback.print_exc()
    return await _resolve_tokenizer(model_id)


async def _resolve_tokenizer(model_id_or_local_path: Union[str, PathLike]):
    """
    内部函数：解析分词器
    
    Args:
        model_id_or_local_path: 模型ID或本地路径
    
    Returns:
        分词器或处理器实例
    
    Raises:
        ValueError: 当模型不支持时抛出
    """
    try:
        # 首先尝试使用AutoProcessor加载
        if DEBUG >= 4: print(f"Trying AutoProcessor for {model_id_or_local_path}")
        processor = AutoProcessor.from_pretrained(
            model_id_or_local_path, 
            use_fast=True if "Mistral-Large" in f"{model_id_or_local_path}" else False,
            trust_remote_code=True
        )
        
        # 为处理器添加必要的属性
        if not hasattr(processor, 'eos_token_id'):
            processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
        if not hasattr(processor, 'encode'):
            processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
        if not hasattr(processor, 'decode'):
            processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
        return processor
    except Exception as e:
        # 处理器加载失败时记录错误
        if DEBUG >= 4: print(f"Failed to load processor for {model_id_or_local_path}. Error: {e}")
        if DEBUG >= 4: print(traceback.format_exc())

    try:
        # 尝试使用AutoTokenizer加载
        if DEBUG >= 4: print(f"Trying AutoTokenizer for {model_id_or_local_path}")
        return AutoTokenizer.from_pretrained(model_id_or_local_path, trust_remote_code=True)
    except Exception as e:
        # Tokenizer加载失败时记录错误
        if DEBUG >= 4: print(f"Failed to load tokenizer for {model_id_or_local_path}. Falling back to tinygrad tokenizer. Error: {e}")
        if DEBUG >= 4: print(traceback.format_exc())

    # 如果所有尝试都失败，抛出错误
    raise ValueError(f"[TODO] Unsupported model: {model_id_or_local_path}")
