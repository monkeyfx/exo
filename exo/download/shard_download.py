from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pathlib import Path
from exo.inference.shard import Shard
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem


class ShardDownloader(ABC):
    """
    分片下载器的抽象基类
    负责管理模型分片的下载过程，确保不会有重复的下载任务
    """
    
    @abstractmethod
    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        """
        确保分片已被下载到本地
        如果尝试下载一个与正在下载的分片重叠的分片，
        当前的下载会被取消，并开始新的下载

        Args:
            shard: 需要下载的分片对象
            inference_engine_name: 托管该分片的推理引擎名称

        Returns:
            Path: 下载文件的本地路径
        """
        pass

    @property
    @abstractmethod
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        """
        下载进度回调系统
        用于通知下载进度事件

        Returns:
            AsyncCallbackSystem: 异步回调系统，处理(分片, 进度事件)元组
        """
        pass


class NoopShardDownloader(ShardDownloader):
    """
    空操作分片下载器
    用于测试或特殊场景，不执行实际的下载操作
    """
    
    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        """
        模拟分片下载
        
        Args:
            shard: 分片对象（未使用）
            inference_engine_name: 推理引擎名称（未使用）

        Returns:
            Path: 固定的临时路径
        """
        return Path("/tmp/noop_shard")

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        """
        提供空的进度回调系统

        Returns:
            AsyncCallbackSystem: 空的异步回调系统
        """
        return AsyncCallbackSystem()
