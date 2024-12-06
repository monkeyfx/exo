from typing import Optional, Tuple, List
import numpy as np
from abc import ABC, abstractmethod
from exo.helpers import AsyncCallbackSystem
from exo.inference.shard import Shard
from exo.topology.topology import Topology


class Node(ABC):
  """
  节点抽象基类，定义了分布式推理系统中节点的基本接口，包括：
1. 节点的生命周期管理（启动/停止）
2. 推理请求处理（文本和张量输入）
3. 结果获取
4. 网络拓扑管理
5. 事件回调系统（token生成和状态更新）
  """
  
  @abstractmethod
  async def start(self, wait_for_peers: int = 0) -> None:
    """
    启动节点
    Args:
        wait_for_peers: 等待连接的对等节点数量
    """
    pass

  @abstractmethod
  async def stop(self) -> None:
    """
    停止节点运行
    """
    pass

  @abstractmethod
  async def process_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.ndarray]:
    """
    处理文本提示
    Args:
        shard: 模型分片
        prompt: 输入的提示文本
        request_id: 请求ID
    Returns:
        处理结果的张量
    """
    pass

  @abstractmethod
  async def process_tensor(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None) -> Optional[np.ndarray]:
    """
    处理张量输入
    Args:
        shard: 模型分片
        tensor: 输入张量
        request_id: 请求ID
    Returns:
        处理结果的张量
    """
    pass

  @abstractmethod
  async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
    """
    获取推理结果
    Args:
        request_id: 请求ID
    Returns:
        (结果张量, 是否完成)的元组
    """
    pass

  @abstractmethod
  async def collect_topology(self, visited: set[str] = set(), max_depth: int = 2) -> Topology:
    """
    收集网络拓扑信息
    Args:
        visited: 已访问的节点集合
        max_depth: 最大搜索深度
    Returns:
        网络拓扑结构
    """
    pass

  @property
  @abstractmethod
  def current_topology(self) -> Topology:
    """
    获取当前节点的拓扑信息
    Returns:
        当前的网络拓扑结构
    """
    pass

  @property
  @abstractmethod
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    """
    Token生成回调系统
    Returns:
        异步回调系统，处理token生成事件
    """
    pass

  @property
  @abstractmethod
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    """
    不透明状态回调系统
    Returns:
        异步回调系统，处理状态更新事件
    """
    pass
