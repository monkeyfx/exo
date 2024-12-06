import numpy as np
import json
import asyncio
import uuid
import time
import traceback
from typing import List, Dict, Optional, Tuple, Union, Set
from exo.networking import Discovery, PeerHandle, Server
from exo.inference.inference_engine import InferenceEngine, Shard
from .node import Node
from exo.topology.topology import Topology
from exo.topology.device_capabilities import device_capabilities
from exo.topology.partitioning_strategy import Partition, PartitioningStrategy, map_partitions_to_shards
from exo import DEBUG
from exo.helpers import AsyncCallbackSystem
from exo.viz.topology_viz import TopologyViz
from exo.download.hf.hf_helpers import RepoProgressEvent
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader

class StandardNode(Node):
  """
  标准节点实现类,负责管理分布式推理的核心功能
  包括:节点发现、模型分片加载、推理处理和结果广播等
  """
  def __init__(
        self,
        _id: str,                    # 节点唯一标识符
        server: Server,              # gRPC服务器实例,用于节点间通信
        inference_engine: InferenceEngine,  # 推理引擎实例(MLX或tinygrad)
        discovery: Discovery,        # 节点发现服务(UDP或Tailscale)
        partitioning_strategy: PartitioningStrategy = None,  # 模型分区策略
        max_generate_tokens: int = 1024,    # 单次生成的最大token数
        default_sample_temperature: float = 0.0,  # token采样温度参数
        topology_viz: Optional[TopologyViz] = None,  # 网络拓扑可视化组件
        shard_downloader: Optional[HFShardDownloader] = None,  # 模型分片下载器
      ):
        # 基础属性初始化
        self.id = _id
        self.inference_engine = inference_engine
        self.server = server
        self.discovery = discovery
        self.partitioning_strategy = partitioning_strategy
        
        # 节点关系和拓扑相关
        self.peers: List[PeerHandle] = {}  # 存储对等节点的连接句柄
        self.topology: Topology = Topology()  # 存储整个网络的拓扑结构
        self.device_capabilities = device_capabilities()  # 获取当前设备的硬件能力信息
        
        # 推理结果缓存
        self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}  # 缓存生成的token序列和完成状态
        self.buffered_logits: Dict[str, List[np.ndarray]] = {}  # 缓存logits输出
        self.buffered_inputs: Dict[str, List[np.ndarray]] = {}  # 缓存输入张量
        
        # 生成参数设置
        self.max_generate_tokens = max_generate_tokens  # 最大生成长度限制
        self.topology_viz = topology_viz  # 拓扑可视化组件
        self.default_sample_temperature = default_sample_temperature  # token采样温度
        
        # 回调系统初始化
        self._on_token = AsyncCallbackSystem[str, Tuple[str, List[int], bool]]()  # token生成事件回调
        self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()  # 状态更新事件回调
        self._on_opaque_status.register("node_status").on_next(self.on_node_status)
        
        # 下载进度追踪
        self.node_download_progress: Dict[str, RepoProgressEvent] = {}  # 记录每个节点的下载进度
        self.topology_inference_engines_pool: List[List[str]] = []  # 记录所有节点支持的推理引擎
        self.shard_downloader = shard_downloader  # 模型分片下载器实例

  async def start(self, wait_for_peers: int = 0) -> None:
    """
    启动节点服务
    
    工作流程:
    1. 启动gRPC服务器
    2. 启动节点发现服务
    3. 等待并更新对等节点
    4. 收集网络拓扑信息
    5. 启动定期拓扑收集任务
    
    Args:
        wait_for_peers: 启动时等待连接的节点数量
    """
    await self.server.start()  # 启动gRPC服务器
    await self.discovery.start()  # 启动节点发现服务
    await self.update_peers(wait_for_peers)  # 更新对等节点列表
    await self.collect_topology()  # 收集初始网络拓扑
    if DEBUG >= 2: print(f"Collected topology: {self.topology}")
    # 创建定期收集拓扑的后台任务
    asyncio.create_task(self.periodic_topology_collection(1.0))

  async def stop(self) -> None:
    """停止节点服务"""
    await self.discovery.stop()  # 停止节点发现服务
    await self.server.stop()  # 停止gRPC服务器

  def on_node_status(self, request_id, opaque_status):
    """
    处理节点状态更新的回调方法
    
    工作流程:
    1. 解析状态数据:
       - 解析JSON格式的状态信息
       - 根据状态类型分别处理
    
    2. 处理不同类型的状态:
       a) 推理引擎支持情况:
          - 记录节点支持的推理引擎类型
          - 更新引擎池信息
          
       b) 节点活动状态:
          - 处理节点开始活动状态
          - 处理节点结束活动状态
          - 更新活动节点标识
          
       c) 下载进度:
          - 解析下载进度事件
          - 更新节点的下载进度记录
    
    3. 更新可视化:
       - 如果存在可视化组件,更新显示
       - 反映最新的拓扑和分区情况
    
    Args:
        request_id: 请求的唯一标识符
        opaque_status: JSON格式的状态信息字符串
        
    Note:
        错误处理会捕获所有异常并记录日志,
        确保状态更新的失败不会影响系统整体运行
    """
    try:
        # 解析状态数据
        status_data = json.loads(opaque_status)
        
        # 处理推理引擎支持情况
        if status_data.get("type", "") == "supported_inference_engines":
            node_id = status_data.get("node_id")
            engines = status_data.get("engines", [])
            self.topology_inference_engines_pool.append(engines)
            
        # 处理节点活动状态
        if status_data.get("type", "") == "node_status":
            # 处理开始状态
            if status_data.get("status", "").startswith("start_"):
                self.current_topology.active_node_id = status_data.get("node_id")
            # 处理结束状态
            elif status_data.get("status", "").startswith("end_"):
                if status_data.get("node_id") == self.current_topology.active_node_id:
                    self.current_topology.active_node_id = None
                    
        # 处理下载进度
        download_progress = None
        if status_data.get("type", "") == "download_progress":
            if DEBUG >= 8: 
                print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
            # 解析并记录下载进度
            download_progress = RepoProgressEvent.from_dict(status_data.get('progress'))
            self.node_download_progress[status_data.get('node_id')] = download_progress
            
        # 更新可视化显示
        if self.topology_viz:
            self.topology_viz.update_visualization(
                self.current_topology,
                self.partitioning_strategy.partition(self.current_topology),
                self.id,
                self.node_download_progress
            )
            
    except Exception as e:
        # 错误处理和日志记录
        if DEBUG >= 1: 
            print(f"Error updating visualization: {e}")
            traceback.print_exc()

  def get_supported_inference_engines(self):
    """
    获取当前节点支持的推理引擎列表
    
    工作流程:
    1. 检查当前推理引擎类型
    2. 根据引擎类型返回支持的引擎列表:
       - MLXDynamicShardInferenceEngine: 支持mlx和tinygrad
       - 其他: 仅支持tinygrad
       
    Returns:
        List[str]: 支持的推理引擎名称列表
        
    Note:
        MLX引擎具有更好的性能,但需要特定硬件支持,
        TinyGrad作为通用后备选项
    """
    supported_engine_names = []
    # MLX引擎同时支持MLX和TinyGrad
    if self.inference_engine.__class__.__name__ == 'MLXDynamicShardInferenceEngine':
        supported_engine_names.append('mlx')
        supported_engine_names.append('tinygrad')
    else:
        # 其他情况只支持TinyGrad
        supported_engine_names.append('tinygrad')
    return supported_engine_names

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    """
    向所有对等节点广播支持的推理引擎信息
    
    工作流程:
    1. 构建包含引擎信息的状态消息
    2. 通过opaque_status机制广播给所有节点
    
    Args:
        supported_engines_names: 支持的推理引擎名称列表
        
    Note:
        使用空request_id广播,因为这是系统级消息,
        不与特定请求关联
    """
    # 构建状态消息
    status_message = json.dumps({
        "type": "supported_inference_engines",
        "node_id": self.id,
        "engines": supported_engines_names
    })
    # 广播状态消息
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    """
    获取网络拓扑中所有节点支持的推理引擎列表
    
    Returns:
        List[List[str]]: 每个节点支持的推理引擎列表的集合
        
    Note:
        此信息用于select_best_inference_engine方法
        选择整个网络中最优的推理引擎
    """
    return self.topology_inference_engines_pool
  
  async def process_inference_result(
    self,
    shard,
    result: np.ndarray,
    request_id: Optional[str] = None,
  ):
    """
    处理推理结果并管理token生成流程
    
    工作流程:
    1. 检查是否达到最大生成长度
    2. 如果是最后一层且未完成:
       - 对结果进行token采样
       - 更新token缓存
       - 检查是否生成了结束标记
       - 广播结果给其他节点
    3. 如果未完成则转发到下一个节点继续处理
    
    Args:
        shard: 当前处理的模型分片
        result: 当前分片的推理结果
        request_id: 用于追踪请求的唯一标识符
    """
    # 初始化结果缓存
    if request_id not in self.buffered_token_output:
        self.buffered_token_output[request_id] = ([], False)
        
    # 检查是否达到最大生成长度
    is_finished = len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
    
    if shard.is_last_layer() and not is_finished:
        # 在最后一层进行token采样
        token = await self.inference_engine.sample(result, temp=self.default_sample_temperature)
        await self.inference_engine.ensure_shard(shard)
        # 更新token缓存
        self.buffered_token_output[request_id][0].append(token.item())
        if DEBUG >= 2: print(f"[{request_id}] result size: {result.size}, is finished: {is_finished}, buffered tokens: {len(self.buffered_token_output[request_id][0])}")
        # 检查是否生成了结束标记
        is_finished = token.item() == self.inference_engine.tokenizer.eos_token_id
        forward = token.reshape(1, -1)
        # 触发回调并广播结果
        self.trigger_on_token_callbacks(request_id, self.buffered_token_output[request_id][0], is_finished)
        asyncio.create_task(self.broadcast_result(request_id, self.buffered_token_output[request_id][0], is_finished))
    else:
      forward = result

    # 更新状态并继续处理
    if is_finished:
      self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
    else:
        # 转发到下一个节点
        asyncio.create_task(self.forward_tensor(shard, forward, request_id, self.get_partition_index(offset = 1)))

    return np.array(self.buffered_token_output[request_id][0])

  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    """
    处理输入提示文本的主要方法
    
    工作流程:
    1. 获取当前节点负责的分片
    2. 广播处理状态
    3. 执行推理处理:
       - 如果是第一层,直接进行推理
       - 如果不是第一层,转发到对应节点处理
    4. 记录处理时间并广播完成状态
    
    Args:
        base_shard: 基础模型分片信息
        prompt: 输入的提示文本
        request_id: 请求的唯一标识符,用于追踪请求状态
        
    Returns:
        Optional[np.ndarray]: 如果是第一层则返回推理结果,否则返回None
    """
    # 获取当前节点负责的分片
    shard = self.get_current_shard(base_shard)
    
    # 广播开始处理状态
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
        }),
      )
    )
    
    # 记录开始时间
    start_time = time.perf_counter_ns()
    
    # 执行实际的处理
    resp = await self._process_prompt(base_shard, prompt, request_id)
    
    # 计算处理耗时
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    
    # 广播完成状态
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
          "result_size": resp.size if resp is not None else 0,
        }),
      )
    )
    return resp

  async def _process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None
  ) -> Optional[np.ndarray]:
    """
    处理提示文本的核心实现
    
    工作流程:
    1. 生成请求ID(如果未提供)
    2. 获取当前节点负责的分片
    3. 根据分片位置决定处理方式:
       - 第一层: 直接进行推理
       - 非第一层: 转发到下一个节点
    
    Args:
        base_shard: 基础模型分片信息
        prompt: 输入的提示文本
        request_id: 请求的唯一标识符
        
    Returns:
        Optional[np.ndarray]: 如果是第一层则返回推理结果,否则返回None
    """
    # 如果没有提供request_id,生成一个新的
    if request_id is None:
        request_id = str(uuid.uuid4())
        
    # 获取当前节点负责的分片
    shard = self.get_current_shard(base_shard)

    if DEBUG >= 2: 
        print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=}")
    
    # 如果不是第一层,转发到下一个分片处理
    if not shard.is_first_layer():
        if DEBUG >= 2: 
            print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=}")
        resp = await self.forward_prompt(shard, prompt, request_id, 0)
        return None
    else:
        # 如果是第一层,进行推理处理
        result = await self.inference_engine.infer_prompt(request_id, shard, prompt)
        ret = await self.process_inference_result(shard, result, request_id) 
        return result

  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    """
    处理输入张量的主要方法
    
    工作流程:
    1. 获取当前节点负责的分片
    2. 广播处理状态
    3. 执行张量推理
    4. 记录处理时间并广播完成状态
    
    Args:
        base_shard: 基础模型分片信息
        tensor: 输入张量数据
        request_id: 请求的唯一标识符
        
    Returns:
        Optional[np.ndarray]: 推理结果张量
    """
    # 获取当前节点负责的分片
    shard = self.get_current_shard(base_shard)
    
    # 广播开始处理状态
    asyncio.create_task(
        self.broadcast_opaque_status(
            request_id,
            json.dumps({
                "type": "node_status",
                "node_id": self.id,
                "status": "start_process_tensor",
                "base_shard": base_shard.to_dict(),
                "shard": shard.to_dict(),
                "tensor_size": tensor.size,
                "tensor_shape": tensor.shape,
                "request_id": request_id,
            }),
        )
    )
    
    # 记录开始时间
    start_time = time.perf_counter_ns()
    # 执行实际的张量处理
    resp = await self._process_tensor(shard, tensor, request_id)
    # 计算处理耗时
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    
    # 广播完成状态
    asyncio.create_task(
        self.broadcast_opaque_status(
            request_id,
            json.dumps({
                "type": "node_status",
                "node_id": self.id,
                "status": "end_process_tensor",
                "base_shard": base_shard.to_dict(),
                "shard": shard.to_dict(),
                "request_id": request_id,
                "elapsed_time_ns": elapsed_time_ns,
                "result_size": resp.size if resp is not None else 0,
            }),
        )
    )
    return resp

  async def _process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    """
    处理张量的核心实现
    
    工作流程:
    1. 生成请求ID(如果未提供)
    2. 获取当前节点负责的分片
    3. 执行张量推理:
       - 调用推理引擎的infer_tensor方法
       - 处理推理结果
    4. 错误处理和日志记录
    
    Args:
        base_shard: 基础模型分片信息
        tensor: 需要处理的输入张量
        request_id: 请求的唯一标识符
        
    Returns:
        Optional[np.ndarray]: 推理结果张量,如果处理失败则返回None
        
    Note:
        此方法包含完整的错误处理,确保即使推理失败也不会影响整体系统运行
    """
    # 如果没有提供request_id,生成一个新的
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    # 获取当前节点负责的分片
    shard = self.get_current_shard(base_shard)

    # 记录输入张量信息
    if DEBUG >= 1: 
        print(f"[{request_id}] process_tensor: {tensor.size=} {tensor.shape=}")
        
    try:
        # 执行张量推理
        result = await self.inference_engine.infer_tensor(request_id, shard, tensor)
        # 处理推理结果
        ret = await self.process_inference_result(shard, result, request_id) 
        return ret
    except Exception as e:
        # 记录错误信息
        print(f"Error processing tensor for shard {shard}: {e}")
        traceback.print_exc()
        return None

  async def forward_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: str,
    target_index: int,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
    if target_id == self.id:
      await self.process_prompt(next_shard, prompt, request_id)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending prompt to {target_peer.id()}: {prompt}")
      await target_peer.send_prompt(next_shard, prompt, request_id=request_id)
  
  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
  ) -> None:
    """
    将张量转发到目标节点进行处理
    
    工作流程:
    1. 根据目标索引获取目标节点ID
    2. 获取目标节点的分片信息
    3. 判断目标节点:
       - 如果是当前节点,直接处理
       - 如果是其他节点,通过peer handle转发
    
    Args:
        base_shard: 基础模型分片信息
        tensor: 需要转发的张量数据
        request_id: 请求的唯一标识符
        target_index: 目标节点的分区索引
        
    Raises:
        ValueError: 当找不到目标peer时抛出异常
    """
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    
    # 获取目标节点ID
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    # 获取目标节点的分片信息
    next_shard = self.get_current_shard(base_shard, target_index)
    
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. target shard: {next_shard}")
    
    # 如果目标是当前节点,直接处理
    if target_id == self.id:
        await self.process_tensor(next_shard, tensor, request_id)
    else:
        # 获取目标节点的peer handle
        target_peer = next((p for p in self.peers if p.id() == target_id), None)
        if not target_peer:
            raise ValueError(f"Peer for {target_index} not found")
            
        if DEBUG >= 1: print(f"Sending tensor to {target_peer.id()}: {tensor}")
        # 通过peer handle发送张量
        await target_peer.send_tensor(next_shard, tensor, request_id=request_id)

  def get_partition_index(self, offset: int = 0):
    """
    获取当前节点在分区中的索引位置
    
    工作流程:
    1. 检查是否存在分区策略
    2. 获取所有分区
    3. 找到当前节点所在的分区索引
    4. 根据偏移量计算目标索引
    
    Args:
        offset: 相对于当前分区的偏移量
        
    Returns:
        int: 目标分区的索引
        
    Raises:
        ValueError: 当找不到当前节点的分区时抛出
    """
    # 检查是否有分区策略
    if not self.partitioning_strategy:
        if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
        return None
        
    # 获取所有分区
    partitions = self.partitioning_strategy.partition(self.topology)
    # 找到当前节点的分区索引
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    
    if current_partition_index is None:
        raise ValueError(f"No current partition found for node: {self.id}")
        
    # 计算目标分区索引(环形结构)
    return (current_partition_index + offset) % len(partitions)

  def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    """
    获取当前节点负责的模型分片
    
    工作流程:
    1. 如果未指定index,获取当前节点的分区索引
    2. 根据分区策略获取所有分区
    3. 将分区映射到具体的模型分片
    4. 返回指定索引的分片
    
    Args:
        base_shard: 原始的完整模型分片信息
        index: 可选的分区索引,如果不指定则使用当前节点的索引
        
    Returns:
        Shard: 当前节点负责的模分片
    """
    # 如果未指定索引,使用当前节点的分区索引
    if index is None:
        index = self.get_partition_index()
        
    # 获取所有分区并映射到分片
    partitions = self.partitioning_strategy.partition(self.topology)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    
    return shards[index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    """
    更新和维护对等节点列表
    
    工作流程:
    1. 发现新的对等节点
    2. 计算节点变化:
       - 新增的节点(peers_added)
       - 移除的节点(peers_removed)
       - 更新的节点(peers_updated)
       - 未变化的节点(peers_unchanged)
    3. 处理连接:
       - 断开需要移除的节点
       - 连接新增和更新的节点
    
    Args:
        wait_for_peers: 等待连接的节点数量
        
    Returns:
        bool: 节点列表是否发生变化
    """
    # 发现新的对等节点
    next_peers = await self.discovery.discover_peers(wait_for_peers)
    
    # 获取当前和新的节点ID集合
    current_peer_ids = {peer.id() for peer in self.peers}
    next_peer_ids = {peer.id() for peer in next_peers}
    
    # 计节点变化
    peers_added = [peer for peer in next_peers if peer.id() not in current_peer_ids]
    peers_removed = [peer for peer in self.peers if peer.id() not in next_peer_ids]
    peers_updated = [peer for peer in next_peers if peer.id() in current_peer_ids and any(p.addr() != peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_unchanged = [peer for peer in next_peers if peer.id() in current_peer_ids and all(p.addr() == peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_to_disconnect = [peer for peer in peers_removed if await peer.is_connected()]
    peers_to_connect = [peer for peer in peers_added + peers_updated + peers_unchanged 
                       if not await peer.is_connected()]

    def _pretty(peers: List[PeerHandle]) -> List[str]:
      return [f"{peer.id()}@{peer.addr()}" for peer in peers]

    if DEBUG >= 2:
      print(f"update_peers: added={peers_added} removed={peers_removed} updated={peers_updated} unchanged={peers_unchanged} to_disconnect={peers_to_disconnect} to_connect={peers_to_connect}")

    async def disconnect_with_timeout(peer, timeout=5):
      """
      在超时限制内断开与对等节点的连接
      
      Args:
          peer: 要断开连接的对等节点
          timeout: 断开连接的超时时间(秒)
          
      Returns:
          bool: 断开连接是否成功
      """
      try:
        await asyncio.wait_for(peer.disconnect(), timeout)
        return True
      except Exception as e:
        print(f"Error disconnecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    async def connect_with_timeout(peer, timeout=5):
      """
      在超时限制内建立与对等节点的连接
      
      工作流程:
      1. 尝试在指定时间内连接对等节点:
         - 使用asyncio.wait_for设置超时限制
         - 调用peer.connect()建立连接
      2. 处理可能的异常情况:
         - 超时异常: 当连接时间超过timeout秒
         - 连接异常: 当网络或目标节点出现问题
      3. 记录连接结果:
         - 成功: 返回True
         - 失败: 打印错误信息并返回False
      
      Args:
          peer: 要连接的对等节点,包含节点ID和地址信息
          timeout: 连接超时时间(秒),默认5秒,防止连接卡死
          
      Returns:
          bool: 连接是否成功建立
          
      Raises:
          不会抛出异常,所有错误都会被捕获并记录
          
      Note:
          - 使用asyncio.wait_for来实现超时控制
          - 连接失败会打印错误信息但不会中断程序运行
          - 通常作为update_peers方法的辅助函数使用
          - 失败的连接会在下一次update_peers时重试
      """
      try:
        # 在timeout秒内尝试建立连接
        await asyncio.wait_for(peer.connect(), timeout)
        return True
      except asyncio.TimeoutError:
        # 连接超时
        print(f"Timeout connecting to peer {peer.id()}@{peer.addr()}")
        return False
      except Exception as e:
        # 其他连接错误
        print(f"Error connecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    # 并行执行所有断开连接的任务
    disconnect_results = await asyncio.gather(
        *(disconnect_with_timeout(peer) for peer in peers_to_disconnect), 
        return_exceptions=True
    )
    
    # 并行执行所有建立连接的任务
    connect_results = await asyncio.gather(
        *(connect_with_timeout(peer) for peer in peers_to_connect), 
        return_exceptions=True
    )

    # 根据执行结果分类处理
    # 成功断开连接的节点
    successful_disconnects = [
        peer for peer, result in zip(peers_to_disconnect, disconnect_results) 
        if result is True
    ]
    # 断开连接失败的节点
    failed_disconnects = [
        peer for peer, result in zip(peers_to_disconnect, disconnect_results) 
        if result is False
    ]
    # 成功建立连接的节点
    successful_connects = [
        peer for peer, result in zip(peers_to_connect, connect_results) 
        if result is True
    ]
    # 建立连接失败的节点
    failed_connects = [
        peer for peer, result in zip(peers_to_connect, connect_results) 
        if result is False
    ]

    # 在调试模式下打印连接状态信息
    if DEBUG >= 1:
        # 打印成功断开连接的节点
        if successful_disconnects: 
            print(f"Successfully disconnected peers: {_pretty(successful_disconnects)}")
        # 打印断开连接失败的节点
        if failed_disconnects: 
            print(f"Failed to disconnect peers: {_pretty(failed_disconnects)}")
        # 打印成功建立连接的节点
        if successful_connects: 
            print(f"Successfully connected peers: {_pretty(successful_connects)}")
        # 打印建立连接失败的节点
        if failed_connects: 
            print(f"Failed to connect peers: {_pretty(failed_connects)}")

    # 更新节点的对等节点列表
    self.peers = next_peers
    # 返回节点列表是否发生变化
    # 只要有新增、移除或更新的节点,就返回True
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  async def select_best_inference_engine(self):
    """
    选择最优的推理引擎
    
    工作流程:
    1. 检查当前推理引擎是否为DummyInferenceEngine
    2. 获取当前节点支持的推理引擎列表
    3. 广播支持的引擎信息给其他节点
    4. 根据拓扑中的引擎信息选择最优引擎
    
    实现细节:
    - 如果当前是DummyInferenceEngine则跳过选择
    - 优先选择第一个支持的引擎(通常是性能最好的)
    - 选择完成后更新当前节点的推理引擎实例
    
    Note:
        此方法用于在分布式环境中协调不同节点使用相同的推理引擎,
        以确保推理结果的一致性
    """
    # 如果是虚拟引擎则跳过
    if self.inference_engine.__class__.__name__ == 'DummyInferenceEngine': 
        return
        
    # 获取支持的引擎列表
    supported_engines = self.get_supported_inference_engines()
    # 广播给其他节点
    await self.broadcast_supported_engines(supported_engines)
    
    # 如果拓扑中有引擎信息,选择第一个支持的引擎
    if len(self.get_topology_inference_engines()):
        self.inference_engine = get_inference_engine(supported_engines[0], self.shard_downloader)

  async def periodic_topology_collection(self, interval: int):
    while True:
      await asyncio.sleep(interval)
      try:
        did_peers_change = await self.update_peers()
        if DEBUG >= 2: print(f"{did_peers_change=}")
        if did_peers_change:
          await self.collect_topology()
          await self.select_best_inference_engine()
      except Exception as e:
        print(f"Error collecting topology: {e}")
        traceback.print_exc()

  async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
    """
    获取指定请求的推理结果
    
    Args:
        request_id: 请求的唯一标识符
        
    Returns:
        Tuple[Optional[np.ndarray], bool]: 
            - 推理结果数组(如果存在)
            - 推理是否完成的标志
    """
    if request_id not in self.buffered_token_output:
        return None, False
    return np.array(self.buffered_token_output[request_id][0]), self.buffered_token_output[request_id][1]

  async def collect_topology(self, visited: set[str] = set(), max_depth: int = 4) -> Topology:
    """
    收集网络拓扑信息
    
    工作流程:
    1. 创建新的拓扑实例并添加当前节点
    2. 递归收集对等节点的拓扑信息:
       - 记录已访问的节点
       - 限制递归深度
       - 合并子拓扑信息
    3. 更新可视化组件(如果存在)
    
    Args:
        visited: 已访问节点的集合,用于避免循环访问
        max_depth: 最大递归深度,防止无限递归
        
    Returns:
        Topology: 收集到的完整网络拓扑
    """
    # 创建新的拓扑实例
    next_topology = Topology()
    next_topology.update_node(self.id, self.device_capabilities)

    if DEBUG >= 2: print(f"Collecting topology {max_depth=} {visited=}")

    # 更新访问记录
    prev_visited = visited.copy()
    visited.add(self.id)
    visited.update(p.id() for p in self.peers)

    # 遍历所有对等节点
    for peer in self.peers:
        # 添加节点和边
        next_topology.update_node(peer.id(), peer.device_capabilities())
        next_topology.add_edge(self.id, peer.id())

        # 跳过已访问的节点
        if peer.id() in prev_visited:
            continue

        # 检查递归深度
        if max_depth <= 0:
            if DEBUG >= 2: print("Max depth reached. Skipping...")
            continue

        # 递归收集子拓扑
        try:
            other_topology = await asyncio.wait_for(
                peer.collect_topology(visited, max_depth=max_depth - 1), 
                timeout=5.0
            )
            if DEBUG >= 2: print(f"Collected topology from: {peer.id()}: {other_topology}")
            self.topology.merge(other_topology)
        except Exception as e:
            print(f"Error collecting topology from {peer.id()}: {e}")
            traceback.print_exc()

    # 更新拓扑信息
    next_topology.active_node_id = self.topology.active_node_id
    self.topology = next_topology
    
    # 更新可视化
    if self.topology_viz:
        self.topology_viz.update_visualization(
            self.current_topology,
            self.partitioning_strategy.partition(self.current_topology),
            self.id
        )
    return next_topology

  @property
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    """
    token生成事件的回调系统
    
    Returns:
        AsyncCallbackSystem: 处理token生成事件的回调系统
        回调函数接收: (request_id, tokens, is_finished)
    """
    return self._on_token

  @property
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    """
    状态更新事件的回调系统
    
    Returns:
        AsyncCallbackSystem: 处理状态更新事件的回调系统
        回调函数接收: (request_id, status)
    """
    return self._on_opaque_status

  def trigger_on_token_callbacks(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
    """
    触发所有token生成事件的回调函数
    
    Args:
        request_id: 请求的唯一标识符
        tokens: 生成的token列表
        is_finished: 生成是否完成的标志
    """
    if DEBUG >= 2: print(f"Triggering all on_token callbacks with {request_id=} num_tokens={len(tokens)} {is_finished=}")
    self.on_token.trigger_all(request_id, tokens, is_finished)
  
  async def broadcast_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    """
    向所有对等节点广播推理结果
    
    工作流程:
    1. 为每个对等节点创建发送任务
    2. 并行执行所有发送任务
    3. 处理超时和错误情况
    
    Args:
        request_id: 请求的唯一标识符
        result: 要广播的token结果列表
        is_finished: 生成是否完成的标志
    """
    async def send_result_to_peer(peer):
        """向单个对等节点发送结果的辅助函数"""
        try:
            await asyncio.wait_for(peer.send_result(request_id, result, is_finished), timeout=15.0)
        except asyncio.TimeoutError:
            print(f"Timeout broadcasting result to {peer.id()}")
        except Exception as e:
            print(f"Error broadcasting result to {peer.id()}: {e}")
            traceback.print_exc()

    # 并行发送给所有对等节点
    await asyncio.gather(*[send_result_to_peer(peer) for peer in self.peers], return_exceptions=True)

  async def broadcast_opaque_status(self, request_id: str, status: str) -> None:
    """
    广播不透明状态信息
    
    工作流程:
    1. 向所有对等节点发送状态信息
    2. 触发本地状态回调
    
    Args:
        request_id: 请求的唯一标识符
        status: 要广播的状态信息(JSON格式字符串)
        
    Note:
        与broadcast_result不同,状态广播会触发本地回调,
        这样当前节点也能接收到状态更新
    """
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")

    async def send_status_to_peer(peer):
        """向单个对等节点发送状态的辅助函数"""
        try:
            await asyncio.wait_for(peer.send_opaque_status(request_id, status), timeout=15.0)
        except asyncio.TimeoutError:
            print(f"Timeout sending opaque status to {peer.id()}")
        except Exception as e:
            print(f"Error sending opaque status to {peer.id()}: {e}")
            traceback.print_exc()

    # 并行发送给所有对等节点
    await asyncio.gather(*[send_status_to_peer(peer) for peer in self.peers], return_exceptions=True)
    # 触发本地态回调
    self.on_opaque_status.trigger_all(request_id, status)

  @property
  def current_topology(self) -> Topology:
    """
    获取当前网络拓扑的只读视图
    
    Returns:
        Topology: 当前的网络拓扑结构
    """
    return self.topology
