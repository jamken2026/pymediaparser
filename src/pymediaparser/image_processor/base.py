"""图像处理器基类定义"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BaseProcessorConfig:
    """处理器配置基类"""

    enabled: bool = True
    """是否启用"""

    fallback_on_error: bool = True
    """处理失败时是否降级为原图"""


class BaseProcessor(ABC):
    """图像处理器基类"""

    def __init__(self, config: BaseProcessorConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """处理器名称"""
        pass

    @abstractmethod
    def should_apply(self, frame_data: Dict[str, Any]) -> bool:
        """判断是否应该对该帧应用此处理器

        Args:
            frame_data: 帧数据字典

        Returns:
            True 表示应用此处理器，False 表示跳过
        """
        pass

    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        """执行图像处理

        Args:
            image: PIL 图像

        Returns:
            处理后的 PIL 图像
        """
        pass

    def process_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理帧数据（包含判断和处理逻辑）

        Args:
            frame_data: 帧数据字典

        Returns:
            处理后的帧数据字典
        """
        if not self.config.enabled:
            return frame_data

        if not self.should_apply(frame_data):
            return frame_data

        try:
            image = frame_data['image']
            processed_image = self.process(image)
            result = frame_data.copy()
            result['image'] = processed_image
            result['preprocessed'] = True
            result['preprocess_strategy'] = self.name
            return result
        except Exception as e:
            logger.warning("预处理失败: %s", e)
            if self.config.fallback_on_error:
                return frame_data
            raise
