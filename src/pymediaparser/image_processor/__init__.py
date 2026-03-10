"""图像处理器模块

提供图像预处理功能，支持多种预处理策略。
"""

from __future__ import annotations

from typing import Union

from .base import BaseProcessor, BaseProcessorConfig
from .resize_processor import ResizeConfig, ResizeProcessor
from .roi_crop_processor import ROICropConfig, ROICropProcessor


def create_processor(
    strategy: str,
    config: Union[ResizeConfig, ROICropConfig, None] = None,
) -> BaseProcessor:
    """创建图像处理器

    Args:
        strategy: 策略名称 'resize' 或 'roi_crop'
        config: 对应策略的配置对象

    Returns:
        处理器实例

    Raises:
        ValueError: 未知的预处理策略
    """
    if strategy == 'resize':
        config = config or ResizeConfig()
        return ResizeProcessor(config)
    elif strategy == 'roi_crop':
        config = config or ROICropConfig()
        return ROICropProcessor(config)
    else:
        raise ValueError(f"未知的预处理策略: {strategy}")


__all__ = [
    # 基类
    'BaseProcessor',
    'BaseProcessorConfig',
    # 缩放处理器
    'ResizeProcessor',
    'ResizeConfig',
    # ROI 裁剪处理器
    'ROICropProcessor',
    'ROICropConfig',
    # 工厂函数
    'create_processor',
]
