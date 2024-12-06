from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from .topology import Topology
from exo.inference.shard import Shard


# 将分片空间划分为连续分片的片段，用0到1之间的浮点数范围[start, end)表示
@dataclass
class Partition:
  node_id: str  # 节点ID
  start: float  # 分区开始位置
  end: float    # 分区结束位置


# 分区策略的抽象基类
class PartitioningStrategy(ABC):
  @abstractmethod
  def partition(self, topology: Topology) -> List[Partition]:
    pass


def map_partitions_to_shards(partitions: List[Partition], num_layers: int, model_id: str) -> List[Shard]:
  """
  将分区映射到具体的分片
  Args:
      partitions: 分区列表
      num_layers: 模型层数
      model_id: 模型ID
  Returns:
      分片列表
  """
  shards = []
  for i, partition in enumerate(partitions):
    # 计算每个分区的起始层和结束层
    start_layer = int(partition.start*num_layers)
    end_layer = int(partition.end*num_layers) - 1

    # 确保最后一个分区覆盖到num_layers - 1
    if i == len(partitions) - 1:
      end_layer = num_layers - 1

    # 确保没有空的分片
    if start_layer <= end_layer:
      shards.append(Shard(model_id, start_layer, end_layer, num_layers))

  # 确保完全覆盖所有层
  # 如果最后一个分片没有覆盖到最后一层，则扩展它
  if shards and shards[-1].end_layer < num_layers - 1:
    shards[-1] = Shard(model_id, shards[-1].start_layer, num_layers - 1, num_layers)

  return shards
