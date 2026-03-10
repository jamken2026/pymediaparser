"""ROI 裁剪处理器 - 对非周期触发帧提取感兴趣区域"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .base import BaseProcessor, BaseProcessorConfig

logger = logging.getLogger(__name__)


@dataclass
class ROICropConfig(BaseProcessorConfig):
    """ROI 裁剪处理器配置"""

    method: str = 'motion'
    """ROI 检测方法: 'motion'(运动检测) | 'saliency'(显著性检测)"""

    padding_ratio: float = 0.2
    """ROI 区域边界扩展比例"""

    min_roi_ratio: float = 0.2
    """最小 ROI 占比阈值，低于此值时扩大到该比例"""


class ROICropProcessor(BaseProcessor):
    """ROI 裁剪处理器 - 仅对非周期触发帧执行"""

    def __init__(self, config: ROICropConfig) -> None:
        super().__init__(config)
        self._roi_config = config
        self._prev_gray: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return 'roi_crop'

    def should_apply(self, frame_data: Dict[str, Any]) -> bool:
        """仅对非周期触发帧应用

        判断逻辑：source 列表只有 'periodic' 时跳过

        Args:
            frame_data: 帧数据字典

        Returns:
            True 表示应用此处理器，False 表示跳过
        """
        source = frame_data.get('source', [])
        # 如果 source 只有 'periodic'，则跳过
        if source == ['periodic']:
            return False
        return True

    def process(self, image: Image.Image) -> Image.Image:
        """执行 ROI 检测并裁剪

        Args:
            image: PIL 图像 (RGB)

        Returns:
            裁剪后的 PIL 图像
        """
        # 转换为 numpy (RGB)
        frame_np = np.array(image)

        # 检测 ROI
        bbox = self._detect_roi(frame_np)

        # 检查最小占比
        bbox = self._ensure_min_ratio(bbox, frame_np.shape)

        # 裁剪
        x, y, w, h = bbox
        cropped = frame_np[y:y+h, x:x+w]

        logger.debug(
            "ROI裁剪: 原图%dx%d -> 裁剪后%dx%d",
            frame_np.shape[1], frame_np.shape[0], w, h,
        )
        return Image.fromarray(cropped)

    def _detect_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """检测 ROI 区域

        Args:
            frame: RGB numpy 数组

        Returns:
            (x, y, w, h) 边界框
        """
        if self._roi_config.method == 'motion':
            return self._detect_by_motion(frame)
        else:
            return self._detect_by_saliency(frame)

    def _detect_by_motion(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """基于帧差法的 ROI 检测

        Args:
            frame: RGB numpy 数组

        Returns:
            (x, y, w, h) 边界框
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return (0, 0, frame.shape[1], frame.shape[0])

        # 帧差
        diff = cv2.absdiff(gray, self._prev_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 轮廓检测
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            self._prev_gray = gray
            return (0, 0, frame.shape[1], frame.shape[0])

        # 合并所有轮廓的边界框
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # 添加 padding
        x, y, w, h = self._add_padding(x, y, w, h, frame.shape)

        self._prev_gray = gray
        return (x, y, w, h)

    def _detect_by_saliency(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """基于显著性检测的 ROI 检测

        Args:
            frame: RGB numpy 数组

        Returns:
            (x, y, w, h) 边界框
        """
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(frame)

        if not success:
            return (0, 0, frame.shape[1], frame.shape[0])

        # 二值化
        _, thresh = cv2.threshold(
            (saliency_map * 255).astype(np.uint8),
            127, 255, cv2.THRESH_BINARY,
        )

        # 轮廓检测 + 取最大轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return (0, 0, frame.shape[1], frame.shape[0])

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        return self._add_padding(x, y, w, h, frame.shape)

    def _add_padding(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """添加边界扩展

        Args:
            x, y, w, h: 原始边界框
            frame_shape: 帧形状 (height, width, channels)

        Returns:
            扩展后的边界框 (x, y, w, h)
        """
        pad_w = int(w * self._roi_config.padding_ratio)
        pad_h = int(h * self._roi_config.padding_ratio)

        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(frame_shape[1] - x, w + 2 * pad_w)
        h = min(frame_shape[0] - y, h + 2 * pad_h)

        return (x, y, w, h)

    def _ensure_min_ratio(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """确保 ROI 占比不低于 min_roi_ratio

        Args:
            bbox: 边界框 (x, y, w, h)
            frame_shape: 帧形状 (height, width, channels)

        Returns:
            调整后的边界框 (x, y, w, h)
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w

        current_area = w * h
        min_area = frame_area * self._roi_config.min_roi_ratio

        if current_area < min_area:
            # 计算需要扩展的比例
            scale = (min_area / current_area) ** 0.5
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 以中心为基准扩展
            cx, cy = x + w // 2, y + h // 2
            new_x = max(0, cx - new_w // 2)
            new_y = max(0, cy - new_h // 2)
            new_w = min(frame_w - new_x, new_w)
            new_h = min(frame_h - new_y, new_h)

            return (new_x, new_y, new_w, new_h)
        return bbox
