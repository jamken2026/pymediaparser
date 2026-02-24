"""帧采样器：按目标频率从解码帧流中抽取帧，并转换为 PIL Image。

核心策略：基于流内时间戳（而非帧计数）控制采样间隔，
适配实时流帧率波动、网络抖动等场景。

典型用法::

    from pymediaparser.frame_sampler import FrameSampler

    sampler = FrameSampler(target_fps=1.0)
    for image, ts in sampler.sample(reader.frames()):
        # image: PIL.Image.Image (RGB), ts: float
        ...
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import av
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FrameSampler:
    """按目标频率从帧流中抽取帧并转换为 PIL Image。

    Args:
        target_fps: 目标输出帧率（帧/秒）。
            例如 ``1.0`` 表示每秒输出一帧，``0.5`` 表示每两秒输出一帧。
    """

    def __init__(self, target_fps: float = 1.0) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps 必须为正数，当前值: {target_fps}")
        self.target_fps = target_fps
        self._interval: float = 1.0 / target_fps  # 采样间隔（秒）
        self._last_emit_ts: float = -float("inf")  # 上次输出帧的流时间戳
        self._frame_count: int = 0  # 已输出帧计数

    @property
    def frame_count(self) -> int:
        """已经输出的帧总数。"""
        return self._frame_count

    def reset(self) -> None:
        """重置采样器内部状态（用于流重连后重新计数等场景）。"""
        self._last_emit_ts = -float("inf")
        self._frame_count = 0

    def sample(
        self,
        frames: Iterator[tuple[av.VideoFrame, float]],
    ) -> Iterator[tuple[Image.Image, float, int]]:
        """从帧流中按时间间隔抽取帧。

        Args:
            frames: 上游帧迭代器，每个元素为 ``(av.VideoFrame, timestamp_seconds)``。

        Yields:
            ``(pil_image, timestamp_seconds, frame_index)`` 三元组。

            - ``pil_image``: RGB 格式的 PIL Image。
            - ``timestamp_seconds``: 帧在流中的时间戳（秒）。
            - ``frame_index``: 该帧在本次采样中的全局序号（从 0 开始）。
        """
        for frame, ts in frames:
            if self._should_emit(ts):
                image = self._frame_to_pil(frame)
                idx = self._frame_count
                self._frame_count += 1
                self._last_emit_ts = ts

                logger.debug(
                    "采样帧 #%d  ts=%.3fs  采样间隔=%.3fs",
                    idx, ts, self._interval,
                )
                yield image, ts, idx

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _should_emit(self, ts: float) -> bool:
        """判断当前帧是否应该输出。

        使用基于流时间戳的策略：当距离上次输出帧的时间间隔
        >= ``1 / target_fps`` 时输出。

        对于时间戳重置/回跳的情况（如流重连），也会触发输出。
        """
        # 时间戳回跳（流重连）=> 立即输出
        if ts < self._last_emit_ts:
            logger.info("检测到时间戳回跳 (%.3f -> %.3f)，重置采样状态", self._last_emit_ts, ts)
            self._last_emit_ts = -float("inf")
            return True
        return (ts - self._last_emit_ts) >= self._interval

    @staticmethod
    def _frame_to_pil(frame: av.VideoFrame) -> Image.Image:
        """将 PyAV 视频帧转换为 RGB PIL Image。"""
        arr: np.ndarray = frame.to_ndarray(format="rgb24")
        return Image.fromarray(arr)
