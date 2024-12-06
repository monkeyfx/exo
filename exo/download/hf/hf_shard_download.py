import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.download.hf.hf_helpers import download_repo_files, RepoProgressEvent, get_weight_map, get_allow_patterns, get_repo_root
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.models import model_cards, get_repo


class HFShardDownloader(ShardDownloader):
    """
    HuggingFace模型分片下载器
    负责从HuggingFace下载模型分片文件并管理下载进度
    """
    def __init__(self, quick_check: bool = False, max_parallel_downloads: int = 4):
        """
        初始化HF分片下载器
        Args:
            quick_check: 是否启用快速检查模式(只检查本地是否存在,不验证完整性)
            max_parallel_downloads: 最大并行下载数量
        """
        self.quick_check = quick_check
        self.max_parallel_downloads = max_parallel_downloads
        self.active_downloads: Dict[Shard, asyncio.Task] = {}  # 当前活跃的下载任务字典
        self.completed_downloads: Dict[Shard, Path] = {}  # 已完成下载的分片路径字典
        self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()  # 下载进度回调系统

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        """
        确保模型分片已下载到本地
        Args:
            shard: 需要下载的模型分片
            inference_engine_name: 推理引擎名称
        Returns:
            Path: 分片文件的本地路径
        """
        repo_name = get_repo(shard.model_id, inference_engine_name)
        
        # 如果分片已经下载完成,直接返回路径
        if shard in self.completed_downloads:
            return self.completed_downloads[shard]
            
        # 快速检查模式:检查本地是否已存在分片
        if self.quick_check:
            repo_root = get_repo_root(repo_name)
            snapshots_dir = repo_root/"snapshots"
            if snapshots_dir.exists():
                visible_dirs = [d for d in snapshots_dir.iterdir() if not d.name.startswith('.')]
                if visible_dirs:
                    most_recent_dir = max(visible_dirs, key=lambda x: x.stat().st_mtime)
                    return most_recent_dir

        # 如果该分片已有下载任务在进行,等待该任务完成
        for active_shard in self.active_downloads:
            if active_shard == shard:
                if DEBUG >= 2: print(f"Download already in progress for {shard}. Keeping that one.")
                return await self.active_downloads[active_shard]

        # 取消同一模型ID下其他分片的下载任务
        existing_active_shards = [active_shard for active_shard in self.active_downloads.keys() if active_shard.model_id == shard.model_id]
        for active_shard in existing_active_shards:
            if DEBUG >= 2: print(f"Cancelling download for {active_shard} (replacing with {shard})")
            task = self.active_downloads[active_shard]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # 取消任务时预期的异常
            except Exception as e:
                if DEBUG >= 2: print(f"Error in cancelling download {active_shard}: {e}")
                traceback.print_exc()
                
        # 清理已取消的下载任务
        self.active_downloads = {active_shard: task for active_shard, task in self.active_downloads.items() 
                               if active_shard.model_id != shard.model_id}

        # 启动新的下载任务
        download_task = asyncio.create_task(self._download_shard(shard, repo_name))
        self.active_downloads[shard] = download_task
        try:
            path = await download_task
            self.completed_downloads[shard] = path
            return path
        finally:
            # 确保任务完成后从活跃下载列表中移除
            print(f"Removing download task for {shard}: {shard in self.active_downloads}")
            if shard in self.active_downloads:
                self.active_downloads.pop(shard)

    async def _download_shard(self, shard: Shard, repo_name: str) -> Path:
        """
        下载指定的模型分片
        Args:
            shard: 需要下载的模型分片
            repo_name: HuggingFace仓库名称
        Returns:
            Path: 下载完成后的本地文件路径
        """
        async def wrapped_progress_callback(event: RepoProgressEvent):
            """包装进度回调函数,触发进度更新事件"""
            self._on_progress.trigger_all(shard, event)

        # 获取权重映射和允许下载的文件模式
        weight_map = await get_weight_map(repo_name)
        allow_patterns = get_allow_patterns(weight_map, shard)

        # 执行实际的文件下载
        return await download_repo_files(repo_name, 
                                       progress_callback=wrapped_progress_callback, 
                                       allow_patterns=allow_patterns, 
                                       max_parallel_downloads=self.max_parallel_downloads)

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        """获取下载进度回调系统"""
        return self._on_progress
