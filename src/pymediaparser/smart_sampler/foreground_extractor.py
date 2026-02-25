"""前景提取器 - 从运动区域提取感兴趣区域"""

from __future__ import annotations
import logging
import cv2
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

class ForegroundExtractor:
    """前景提取器 - 基于运动掩码提取前景区域"""

    def __init__(self, padding_ratio: float = 0.2) -> None:
        """初始化前景提取器。
        
        Args:
            padding_ratio: 边缘扩展比例，相对于检测区域尺寸。
                          例如 0.2 表示在四边各扩展检测区域宽/高的 20%。
        """
        self.padding_ratio = padding_ratio
        logger.debug("ForegroundExtractor 初始化完成 - 边缘扩展比例: %.1f%%", padding_ratio * 100)

    def extract_foreground(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if frame is None or mask is None:
            return frame, (0, 0, frame.shape[1], frame.shape[0]) if frame is not None else (0, 0, 0, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.debug("未检测到运动区域，返回原图")
            return frame, (0, 0, frame.shape[1], frame.shape[0])
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)
        
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(frame.shape[1] - x, w + 2 * pad_x)
        h = min(frame.shape[0] - y, h + 2 * pad_y)
        
        cropped = frame[y:y+h, x:x+w]
        
        logger.debug("前景提取完成 - 裁剪后尺寸: %dx%d", cropped.shape[1], cropped.shape[0])
        return cropped, (x, y, w, h)

    def calculate_compression_ratio(self, original_shape: tuple, cropped_shape: tuple) -> float:
        original_pixels = original_shape[0] * original_shape[1]
        cropped_pixels = cropped_shape[0] * cropped_shape[1]
        
        if original_pixels > 0:
            ratio = cropped_pixels / original_pixels
            logger.debug("图像压缩比率: %.2f%%", ratio * 100)
            return ratio
        return 1.0
