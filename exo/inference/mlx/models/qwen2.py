from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen2 import TransformerBlock, ModelArgs

from ...shard import Shard
from .base import IdentityBlock


@dataclass
class ModelArgs(ModelArgs):
    """
    模型参数配置类，继承自基础ModelArgs
    添加了分片相关的配置
    """
    # 使用field默认工厂函数初始化分片配置
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        """
        初始化后的处理
        验证分片配置的正确性
        """
        super().__post_init__()  # 确保父类初始化被执行

        if isinstance(self.shard, Shard):
            return
        if not isinstance(self.shard, dict):
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

        self.shard = Shard(**self.shard)


class Qwen2Model(nn.Module):
    """
    Qwen2模型的核心实现类
    该代码实现了Qwen2模型的分布式版本，具有以下主要特点：
    - **模型分片**：
      - 可以将模型划分为多个部分，在不同设备上运行。
      - 通过`Shard`类控制每个分片的层范围。
    - **模块化设计**：
      - `ModelArgs`类负责处理配置参数。
      - `Qwen2Model`类实现核心Transformer结构。
      - `Model`类提供完整的模型封装。
    - **优化特性**：
      - 支持注意力缓存。
      - 支持权重共享。
      - 提供权重过滤机制。
    - **灵活性**：
      - 可配置的模型参数。
      - 可选的权重共享。
      - 支持不同的运行模式。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化Qwen2模型
        Args:
            args: 模型参数配置
        """
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0

        # 只在第一个分片中初始化词嵌入层
        if self.args.shard.is_first_layer():
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        # 初始化Transformer层
        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.args.shard.start_layer <= i <= self.args.shard.end_layer:
                self.layers.append(TransformerBlock(args=args))
            else:
                self.layers.append(IdentityBlock())

        # 只在最后一个分片中初始化归一化层
        if self.args.shard.is_last_layer():
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        """
        模型前向传播
        Args:
            inputs: 输入张量
            cache: 注意力缓存
        Returns:
            模型输出
        """
        if self.args.shard.is_first_layer():
            h = self.embed_tokens(inputs)
        else:
            h = inputs

        # 创建注意力掩码
        mask = None
        if h.shape[1] > 1:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None]*len(self.layers)

        # 通过所有Transformer层
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        if self.args.shard.is_last_layer():
            h = self.norm(h)
        return h


class Model(nn.Module):
    """
    完整的Qwen2模型封装类
    包含主干网络和语言模型头
    """
    def __init__(self, args: ModelArgs):
        """
        初始化模型
        Args:
            args: 模型参数配置
        """
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen2Model(args)
        
        # 只在最后一个分片中初始化语言模型头
        if self.args.shard.is_last_layer():
            if not args.tie_word_embeddings:
                self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        """
        模型前向传播
        Args:
            inputs: 输入张量
            cache: 注意力缓存
        Returns:
            模型输出
        """
        out = self.model(inputs, cache)
        if self.args.shard.is_last_layer():
            if self.args.tie_word_embeddings:
                out = self.model.embed_tokens.as_linear(out)
            else:
                out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        """
        清理和过滤权重
        只保留当前分片需要的权重
        Args:
            weights: 原始权重字典
        Returns:
            过滤后的权重字典
        """
        shard_state_dict = {}

        for key, value in weights.items():
            # 跳过位置编码权重
            if "self_attn.rotary_emb.inv_freq" in key:
                continue
            
            # 处理Transformer层的权重
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])
                if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
                    shard_state_dict[key] = value
            
            # 处理第一个分片的词嵌入层权重
            elif self.args.shard.is_first_layer() and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
            
            # 处理最后一个分片的词嵌入层权重（如果使用权重共享）
            elif (self.args.shard.is_last_layer() and self.args.tie_word_embeddings) and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
            
            # 处理最后一个分片的语言模型头权重（如果不使用权重共享）
            elif (self.args.shard.is_last_layer() and not self.args.tie_word_embeddings) and key.startswith('lm_head'):
                shard_state_dict[key] = value
            
            # 处理最后一个分片的归一化层权重
            elif self.args.shard.is_last_layer() and (key.startswith('model.norm')):
                shard_state_dict[key] = value

        # 如果使用权重共享，移除语言模型头的权重
        if self.args.tie_word_embeddings:
            shard_state_dict.pop("lm_head.weight", None)

        return shard_state_dict

    @property
    def layers(self):
        """获取模型的所有层"""
        return self.model.layers

    @property
    def head_dim(self):
        """获取注意力头的维度"""
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        """获取键值注意力头的数量"""
        return self.args.num_key_value_heads
