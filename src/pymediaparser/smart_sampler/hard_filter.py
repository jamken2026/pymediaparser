"""硬性过滤器 (Layer 0) - 快速排除绝对不需要处理的帧"""

from __future__ import annotations
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 下采样目标尺寸，用于加速计算
_SMALL_WIDTH = 160
_SMALL_HEIGHT = 90


class HardFilter:
    """Layer 0 硬性过滤器 - 纯逻辑判断，无复杂计算，目标耗时 <0.1ms。

    过滤项（按顺序执行，任一命中即拒绝）：
    1. 时间间隔控制 - 强制最小采样间隔
    2. 静止帧检测 - 帧间差分法
    3. 画面异常检测 - 黑屏/白屏/花屏
    4. 熵异常检测 - 直方图熵值异常
    5. 分辨率一致性 - 尺寸突变检测
    """

    def __init__(
        self,
        min_frame_interval: float = 1.0,
        still_threshold: float = 2.0,
        black_threshold: int = 10,
        white_threshold: int = 245,
        entropy_low: float = 3.0,
        entropy_high: float = 7.5,
        size_change_tolerance: float = 0.05,
    ) -> None:
        self._min_frame_interval = min_frame_interval
        self._still_threshold = still_threshold
        self._black_threshold = black_threshold
        self._white_threshold = white_threshold
        self._entropy_low = entropy_low
        self._entropy_high = entropy_high
        self._size_tolerance = size_change_tolerance

        # 状态
        self._last_accept_ts: float = -float('inf')
        self._prev_gray_small: Optional[np.ndarray] = None
        self._prev_shape: Optional[Tuple[int, int]] = None  # (h, w)
        self._reject_count: int = 0
        self._accept_count: int = 0

        logger.debug(
            "HardFilter 初始化完成 - 最小间隔: %.1fs, 静止阈值: %.1f, "
            "黑屏: <%d, 白屏: >%d, 熵: [%.1f, %.1f]",
            min_frame_interval, still_threshold,
            black_threshold, white_threshold,
            entropy_low, entropy_high,
        )

    def check(self, frame: np.ndarray, timestamp: float) -> Tuple[bool, str]:
        """检查帧是否通过硬性过滤。

        Args:
            frame: BGR 格式图像帧。
            timestamp: 帧时间戳（秒）。

        Returns:
            (passed, reason): passed 为 True 表示通过，reason 为拒绝原因（通过时为空字符串）。
        """
        # 1. 时间间隔控制
        if timestamp - self._last_accept_ts < self._min_frame_interval:
            self._reject_count += 1
            return False, '时间间隔'

        # 转换灰度 + 下采样
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (_SMALL_WIDTH, _SMALL_HEIGHT), interpolation=cv2.INTER_NEAREST)

        # 2. 静止帧检测
        if self._prev_gray_small is not None:
            diff = cv2.absdiff(gray_small, self._prev_gray_small)
            mean_diff = float(np.mean(np.asarray(diff)))  # type: ignore[arg-type]
            if mean_diff < self._still_threshold:
                self._reject_count += 1
                # 不更新 _prev_gray_small，保持与上一个非静止帧比较
                return False, '静止帧'

        # 更新缓存的灰度小图
        self._prev_gray_small = gray_small

        # 3. 黑屏/白屏检测
        mean_val = float(np.mean(np.asarray(gray_small)))  # type: ignore[arg-type]
        if mean_val < self._black_threshold:
            self._reject_count += 1
            return False, '黑屏'
        if mean_val > self._white_threshold:
            self._reject_count += 1
            return False, '白屏'

        # 4. 熵异常检测（直方图熵）
        hist = cv2.calcHist([gray_small], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        # 避免 log2(0)
        nonzero = hist[hist > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        if entropy < self._entropy_low:
            self._reject_count += 1
            logger.debug("熵值过低: %.2f < %.2f (画面单调)", entropy, self._entropy_low)
            return False, f'熵异常(过低:{entropy:.2f})'
        if entropy > self._entropy_high:
            self._reject_count += 1
            logger.debug("熵值过高: %.2f > %.2f (疑似噪声)", entropy, self._entropy_high)
            return False, f'熵异常(过高:{entropy:.2f})'

        # 5. 分辨率一致性
        h, w = frame.shape[:2]
        if self._prev_shape is not None:
            prev_pixels = self._prev_shape[0] * self._prev_shape[1]
            curr_pixels = h * w
            if prev_pixels > 0:
                change_ratio = abs(curr_pixels - prev_pixels) / prev_pixels
                if change_ratio > self._size_tolerance:
                    self._reject_count += 1
                    return False, '分辨率突变'
        self._prev_shape = (h, w)

        # 通过
        self._last_accept_ts = timestamp
        self._accept_count += 1
        return True, ''

    def reset(self) -> None:
        """重置过滤器状态。"""
        self._last_accept_ts = -float('inf')
        self._prev_gray_small = None
        self._prev_shape = None
        self._reject_count = 0
        self._accept_count = 0
        logger.info("HardFilter 状态已重置")

    @property
    def reject_count(self) -> int:
        """累计拒绝帧数。"""
        return self._reject_count

    @property
    def accept_count(self) -> int:
        """累计通过帧数。"""
        return self._accept_count

    @property
    def pass_rate(self) -> float:
        """累计通过率（0~1）。"""
        total = self._accept_count + self._reject_count
        return self._accept_count / total if total > 0 else 0.0
