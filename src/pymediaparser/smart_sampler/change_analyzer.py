"""变化分析器 - 多维度帧间变化检测"""

from __future__ import annotations
import logging
import cv2
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)

class ChangeAnalyzer:
    """变化分析器 - 综合多种方法分析帧间变化"""

    def __init__(self, ssim_threshold: float = 0.85) -> None:
        self.ssim_threshold = ssim_threshold
        self.reference_frame: Optional[np.ndarray] = None
        self.reference_gray: Optional[np.ndarray] = None
        logger.debug("ChangeAnalyzer 初始化完成")

    def analyze_change(self, current_frame: np.ndarray) -> Dict[str, float]:
        """分析帧间变化。"""
        if current_frame is None or current_frame.size == 0:
            return self._empty_result()
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if self.reference_frame is None:
            self._set_reference(current_frame, current_gray)
            return self._no_change_result()
        
        ssim_score = self._simple_similarity(current_gray)
        significant = ssim_score < self.ssim_threshold
        
        result = {
            'significant': significant,
            'ssim_score': ssim_score,
            'combined_score': ssim_score,
        }
        
        logger.debug("变化分析 - 分数: %.3f, 显著变化: %s", ssim_score, significant)
        return result
    
    def update_reference(self, frame: np.ndarray) -> None:
        """更新参考帧为指定帧（采样成功后调用）。"""
        if frame is not None and frame.size > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._set_reference(frame, gray)
            logger.debug("参考帧已更新为采样帧")

    def _simple_similarity(self, current_gray: np.ndarray) -> float:
        """使用SSIM算法计算结构相似度"""
        if self.reference_gray is None:
            return 1.0
        
        try:
            from skimage.metrics import structural_similarity as ssim
            # 计算SSIM，返回相似度分数（0-1，1表示完全相同）
            score = ssim(self.reference_gray, current_gray)
            return float(score)  # type: ignore[arg-type]  # ssim 默认返回 float，仅当 full=True 时返回元组
        except ImportError:
            # 如果没有skimage，使用简化的MSE-based计算
            mse = np.mean((current_gray.astype(float) - self.reference_gray.astype(float)) ** 2)
            similarity = 1.0 / (1.0 + mse / 10000.0)
            return max(0.0, min(1.0, float(similarity)))

    def _set_reference(self, frame: np.ndarray, gray_frame: np.ndarray) -> None:
        self.reference_frame = frame.copy()
        self.reference_gray = gray_frame.copy()
        logger.debug("参考帧已设置")

    def _no_change_result(self) -> Dict[str, float]:
        return {
            'significant': False,
            'ssim_score': 1.0,
            'combined_score': 1.0,
        }

    def _empty_result(self) -> Dict[str, float]:
        return {
            'significant': False,
            'ssim_score': 0.0,
            'combined_score': 0.0,
        }

    def reset(self) -> None:
        self.reference_frame = None
        self.reference_gray = None
        logger.info("变化分析器已重置")
