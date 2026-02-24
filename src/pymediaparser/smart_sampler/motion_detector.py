"""运动检测器 - 基于OpenCV背景减除算法"""

from __future__ import annotations
import logging
import cv2
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

class MotionDetector:
    """运动检测器 - 使用背景减除算法检测帧中的运动区域"""

    def __init__(self, method: str = 'MOG2', learning_rate: float = 0.01, threshold: float = 0.01) -> None:
        self.method = method.upper()
        self.learning_rate = learning_rate
        self.threshold = threshold  # 运动检测阈值（运动像素比例）
        self.bg_subtractor = self._create_bg_subtractor()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        logger.debug("MotionDetector 初始化完成 - 阈值: %.3f", threshold)

    def _create_bg_subtractor(self):
        if self.method == 'MOG2':
            return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        elif self.method == 'KNN':
            return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
        else:
            raise ValueError(f"不支持的方法: {self.method}")

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        if frame is None or frame.size == 0:
            return False, 0.0, np.zeros((1, 1), dtype=np.uint8)
        
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_score = motion_pixels / total_pixels if total_pixels > 0 else 0.0
        
        has_motion = motion_score > self.threshold
        logger.debug("运动检测 - 像素变化: %.2f%% (阈值: %.2f%%), 有运动: %s", 
                    motion_score * 100, self.threshold * 100, has_motion)
        
        return has_motion, motion_score, fg_mask

    def reset(self) -> None:
        self.bg_subtractor = self._create_bg_subtractor()
        logger.info("运动检测器已重置")
