from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
    """
    基于环形拓扑的内存加权分区策略
    
    该策略根据每个节点的可用内存大小,按比例分配模型层。
    节点按环形排列,每个节点负责处理其分配到的连续层。
    """
    
    def partition(self, topology: Topology) -> List[Partition]:
        """
        将模型分区到不同节点
        
        Args:
            topology: 当前的网络拓扑结构,包含所有节点信息
            
        Returns:
            List[Partition]: 分区列表,每个分区包含节点ID和其负责的层范围
            
        工作流程:
        1. 获取所有节点并按内存大小降序排序
        2. 计算总内存
        3. 根据内存比例计算每个节点应分配的层范围
        """
        # 获取所有节点并按内存大小和节点ID排序(内存大的优先)
        nodes = list(topology.all_nodes())
        nodes.sort(key=lambda x: (x[1].memory, x[0]), reverse=True)
        
        # 计算所有节点的总内存
        total_memory = sum(node[1].memory for node in nodes)
        
        # 用于存储分区结果
        partitions = []
        start = 0
        
        # 为每个节点分配分区
        for node in nodes:
            # 根据节点内存占比计算结束位置
            # round()用于处理浮点数精度问题
            end = round(start + (node[1].memory/total_memory), 5)
            
            # 创建分区并添加到结果列表
            # node[0]是节点ID
            # start和end表示该节点负责的层范围(0-1之间的比例)
            partitions.append(Partition(node[0], start, end))
            
            # 更新下一个分区的起始位置
            start = end
            
        return partitions
