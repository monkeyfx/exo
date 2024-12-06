from typing import Dict, Callable, Coroutine, Any, Literal
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class RepoFileProgressEvent:
    """
    仓库文件下载进度事件
    记录单个文件的下载状态和进度信息
    """
    repo_id: str          # 仓库ID
    repo_revision: str    # 仓库版本
    file_path: str        # 文件路径
    downloaded: int       # 已下载的字节数
    downloaded_this_session: int  # 本次会话下载的字节数
    total: int           # 文件总字节数
    speed: int           # 下载速度（字节/秒）
    eta: timedelta       # 预计剩余时间
    status: Literal["not_started", "in_progress", "complete"]  # 下载状态

    def to_dict(self):
        """
        将事件对象转换为字典格式
        Returns:
            dict: 包含所有字段的字典
        """
        return {
            "repo_id": self.repo_id,
            "repo_revision": self.repo_revision,
            "file_path": self.file_path,
            "downloaded": self.downloaded,
            "downloaded_this_session": self.downloaded_this_session,
            "total": self.total,
            "speed": self.speed,
            "eta": self.eta.total_seconds(),
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data):
        """
        从字典创建事件对象
        Args:
            data: 包含所有必要字段的字典
        Returns:
            RepoFileProgressEvent: 新的事件对象
        """
        if 'eta' in data:
            data['eta'] = timedelta(seconds=data['eta'])
        return cls(**data)


@dataclass
class RepoProgressEvent:
    """
    仓库整体下载进度事件
    记录整个仓库的下载状态和进度信息
    """
    repo_id: str          # 仓库ID
    repo_revision: str    # 仓库版本
    completed_files: int  # 已完成的文件数
    total_files: int      # 总文件数
    downloaded_bytes: int # 已下载的总字节数
    downloaded_bytes_this_session: int  # 本次会话下载的字节数
    total_bytes: int      # 总字节数
    overall_speed: int    # 整体下载速度（字节/秒）
    overall_eta: timedelta  # 预计剩余时间
    file_progress: Dict[str, RepoFileProgressEvent]  # 各文件的进度信息
    status: Literal["not_started", "in_progress", "complete"]  # 下载状态

    def to_dict(self):
        """
        将事件对象转换为字典格式
        Returns:
            dict: 包含所有字段的字典
        """
        return {
            "repo_id": self.repo_id,
            "repo_revision": self.repo_revision,
            "completed_files": self.completed_files,
            "total_files": self.total_files,
            "downloaded_bytes": self.downloaded_bytes,
            "downloaded_bytes_this_session": self.downloaded_bytes_this_session,
            "total_bytes": self.total_bytes,
            "overall_speed": self.overall_speed,
            "overall_eta": self.overall_eta.total_seconds(),
            "file_progress": {k: v.to_dict() for k, v in self.file_progress.items()},
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data):
        """
        从字典创建事件对象
        Args:
            data: 包含所有必要字段的字典
        Returns:
            RepoProgressEvent: 新的事件对象
        """
        if 'overall_eta' in data:
            data['overall_eta'] = timedelta(seconds=data['overall_eta'])
        if 'file_progress' in data:
            data['file_progress'] = {
                k: RepoFileProgressEvent.from_dict(v)
                for k, v in data['file_progress'].items()
            }
        return cls(**data)


# 类型别名定义
RepoFileProgressCallback = Callable[[RepoFileProgressEvent], Coroutine[Any, Any, None]]
RepoProgressCallback = Callable[[RepoProgressEvent], Coroutine[Any, Any, None]]
