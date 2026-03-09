"""Pipeline 基础模块：状态定义、进度信息、抽象接口。

提供 LivePipeline 和 ReplayPipeline 的公共类型定义：
- PipelineState: 运行状态枚举
- PipelineProgress: 进度信息数据类
- BasePipeline: 抽象接口定义

状态契约
--------
进入任何终态（COMPLETED/STOPPED/ERROR）时，资源已清理完毕。
调用者无需在终态时调用 stop()。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .vlm_base import StreamConfig, VLMClient
    from .result_handler import ResultHandler


class PipelineState(Enum):
    """Pipeline 运行状态。

    状态流转：
        IDLE → STARTING → RUNNING → STOPPING → STOPPED
                            │
                            ├── 正常完成 → COMPLETED
                            │
                            └── 异常 → ERROR

    终态（COMPLETED/STOPPED/ERROR）时资源已清理。
    """

    IDLE = "idle"               # 初始状态，未启动
    STARTING = "starting"       # 启动中（加载模型）
    RUNNING = "running"         # 正常运行中
    STOPPING = "stopping"       # 停止中（清理资源）
    COMPLETED = "completed"     # 正常完成（仅 ReplayPipeline）
    STOPPED = "stopped"         # 被主动停止
    ERROR = "error"             # 异常终止


@dataclass
class PipelineProgress:
    """Pipeline 执行进度信息。

    Attributes:
        state: 当前运行状态
        processed_frames: 已处理帧数
        start_time: 启动时间戳（time.time()）
        total_frames: 总帧数（仅 ReplayPipeline 可获取）
        current_timestamp: 当前处理到的时间戳（视频内时间）
        duration: 视频总时长（仅 ReplayPipeline 可获取）
        error: 异常信息（仅 ERROR 状态时有值）
        extra: 扩展信息字典
    """
    state: PipelineState
    processed_frames: int = 0
    start_time: float = 0.0
    total_frames: Optional[int] = None
    current_timestamp: float = 0.0
    duration: Optional[float] = None
    error: Optional[Exception] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_time(self) -> float:
        """已运行时长（秒）。"""
        if self.start_time == 0.0:
            return 0.0
        return time.time() - self.start_time

    @property
    def progress_percent(self) -> Optional[float]:
        """进度百分比（仅 ReplayPipeline 有意义）。

        Returns:
            0-100 的百分比，或 None（无法计算时）
        """
        if self.duration is None or self.duration <= 0:
            return None
        return min(100.0, (self.current_timestamp / self.duration) * 100.0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            "state": self.state.value,
            "processed_frames": self.processed_frames,
            "start_time": self.start_time,
            "elapsed_time": self.elapsed_time,
            "total_frames": self.total_frames,
            "current_timestamp": self.current_timestamp,
            "duration": self.duration,
            "progress_percent": self.progress_percent,
            "error": str(self.error) if self.error else None,
            "extra": self.extra,
        }


class BasePipeline(ABC):
    """Pipeline 抽象基类（纯接口定义）。

    定义 Pipeline 的公共接口，具体实现由子类完成。

    状态契约：
        进入任何终态时，资源已清理完毕。
    """

    @abstractmethod
    def start(self) -> None:
        """启动 Pipeline（非阻塞）。

        状态转换：IDLE → STARTING → RUNNING
        启动失败时转换为 ERROR。
        """

    @abstractmethod
    def stop(self) -> None:
        """停止 Pipeline（幂等）。

        仅在 RUNNING 或 STARTING 状态时执行停止逻辑。
        状态转换：RUNNING/STARTING → STOPPING → STOPPED
        """

    @abstractmethod
    def run(self) -> None:
        """阻塞式运行，支持 Ctrl+C 优雅退出。"""

    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待 Pipeline 进入终态。

        Args:
            timeout: 超时秒数，None 表示无限等待

        Returns:
            True 表示正常进入终态，False 表示超时
        """

    @abstractmethod
    def get_state(self) -> PipelineState:
        """获取当前运行状态。"""

    @abstractmethod
    def get_progress(self) -> PipelineProgress:
        """获取进度信息。"""

    @abstractmethod
    def is_running(self) -> bool:
        """是否运行中（RUNNING 状态）。"""

    @abstractmethod
    def is_completed(self) -> bool:
        """是否正常完成（COMPLETED 状态）。"""

    @abstractmethod
    def is_stopped(self) -> bool:
        """是否被主动停止（STOPPED 状态）。"""

    @abstractmethod
    def is_error(self) -> bool:
        """是否异常终止（ERROR 状态）。"""

    @abstractmethod
    def is_terminal(self) -> bool:
        """是否处于终态（可重新启动）。"""

    @staticmethod
    def _is_terminal_state(state: PipelineState) -> bool:
        """判断是否为终态。"""
        return state in (
            PipelineState.COMPLETED,
            PipelineState.STOPPED,
            PipelineState.ERROR,
        )
