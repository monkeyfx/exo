from .device_capabilities import DeviceCapabilities
from typing import Dict, Set, Optional


class Topology:
  """
  网络拓扑类，管理节点和它们之间的连接关系
  定义了一个Topology类，用于管理网络中的节点及其连接关系。主要功能包括：
  1. 添加和更新节点及其设备能力
  2. 管理节点之间的连接
  3. 合并其他拓扑
  4. 提供拓扑的字符串表示以便于调试和日志记录
  """
  
  def __init__(self):
    # 节点字典，映射节点ID到设备能力
    self.nodes: Dict[str, DeviceCapabilities] = {}
    # 邻接表，表示节点之间的连接关系
    self.peer_graph: Dict[str, Set[str]] = {}
    # 当前活跃节点的ID
    self.active_node_id: Optional[str] = None

  def update_node(self, node_id: str, device_capabilities: DeviceCapabilities):
    """
    更新或添加节点的设备能力
    Args:
        node_id: 节点ID
        device_capabilities: 设备能力
    """
    self.nodes[node_id] = device_capabilities

  def get_node(self, node_id: str) -> DeviceCapabilities:
    """
    获取节点的设备能力
    Args:
        node_id: 节点ID
    Returns:
        设备能力
    """
    return self.nodes.get(node_id)

  def all_nodes(self):
    """
    获取所有节点及其设备能力
    Returns:
        节点ID和设备能力的字典项
    """
    return self.nodes.items()

  def add_edge(self, node1_id: str, node2_id: str):
    """
    添加两个节点之间的连接
    Args:
        node1_id: 第一个节点ID
        node2_id: 第二个节点ID
    """
    if node1_id not in self.peer_graph:
      self.peer_graph[node1_id] = set()
    if node2_id not in self.peer_graph:
      self.peer_graph[node2_id] = set()
    self.peer_graph[node1_id].add(node2_id)
    self.peer_graph[node2_id].add(node1_id)

  def get_neighbors(self, node_id: str) -> Set[str]:
    """
    获取指定节点的邻居节点
    Args:
        node_id: 节点ID
    Returns:
        邻居节点ID集合
    """
    return self.peer_graph.get(node_id, set())

  def all_edges(self):
    """
    获取所有节点之间的连接
    Returns:
        节点连接的列表
    """
    edges = []
    for node, neighbors in self.peer_graph.items():
      for neighbor in neighbors:
        if (neighbor, node) not in edges:  # 避免重复边
          edges.append((node, neighbor))
    return edges

  def merge(self, other: "Topology"):
    """
    合并另一个拓扑到当前拓扑
    Args:
        other: 另一个拓扑对象
    """
    for node_id, capabilities in other.nodes.items():
      self.update_node(node_id, capabilities)
    for node_id, neighbors in other.peer_graph.items():
      for neighbor in neighbors:
        self.add_edge(node_id, neighbor)

  def __str__(self):
    """
    返回拓扑的字符串表示
    Returns:
        拓扑的字符串
    """
    nodes_str = ", ".join(f"{node_id}: {cap}" for node_id, cap in self.nodes.items())
    edges_str = ", ".join(f"{node}: {neighbors}" for node, neighbors in self.peer_graph.items())
    return f"Topology(Nodes: {{{nodes_str}}}, Edges: {{{edges_str}}})"
