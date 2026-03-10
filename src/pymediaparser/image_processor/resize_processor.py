"""缩放处理器 - 对所有帧执行等比缩放"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from PIL import Image

from .base import BaseProcessor, BaseProcessorConfig

logger = logging.getLogger(__name__)


@dataclass
class ResizeConfig(BaseProcessorConfig):
    """缩放处理器配置"""

    max_size: int = 1024
    """图像最大边长（像素），超过时等比缩放"""


class ResizeProcessor(BaseProcessor):
    """缩放处理器 - 对所有帧执行等比缩放"""

    def __init__(self, config: ResizeConfig) -> None:
        super().__init__(config)
        self._resize_config = config

    @property
    def name(self) -> str:
        return 'resize'

    def should_apply(self, frame_data: Dict[str, Any]) -> bool:
        """对所有帧都应用"""
        return True

    def process(self, image: Image.Image) -> Image.Image:
        """等比缩放图像

        Args:
            image: PIL 图像

        Returns:
            缩放后的 PIL 图像
        """
        w, h = image.size
        max_size = self._resize_config.max_size

        # 如果图像已经小于等于目标尺寸，直接返回
        if max(w, h) <= max_size:
            return image

        # 计算等比缩放后的尺寸
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)

        logger.debug("缩放图像: %dx%d -> %dx%d", w, h, new_w, new_h)
        return image.resize((new_w, new_h), Image.LANCZOS)
