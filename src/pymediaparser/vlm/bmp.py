"""BMP虚拟VLM后端：用于调试分析的图像保存后端。

该后端不实际调用大模型进行帧分析，而是将收到的图像保存为BMP文件。
主要用于调试、验证帧采集流程、检查图像质量等场景。

典型用法::

    from pymediaparser.vlm import create_vlm_client
    from pymediaparser.vlm_base import VLMConfig

    config = VLMConfig(model_path="/tmp/debug_frames")
    client = create_vlm_client("bmp", config)

    with client:
        result = client.analyze(pil_image, "调试帧")
        # result.text = '{"files": ["/tmp/debug_frames/frame_000001.bmp"]}'
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import List, Sequence

from PIL import Image

from ..vlm_base import VLMClient, VLMConfig, VLMResult

logger = logging.getLogger(__name__)


class BMPVLMClient(VLMClient):
    """BMP虚拟VLM后端客户端。

    将图像保存为BMP文件，不进行实际的VLM推理。
    适用于调试分析、帧采集验证等场景。

    Args:
        config: VLM配置，其中 model_path 用作BMP文件保存目录。

    Attributes:
        config.model_path: BMP文件保存目录路径。
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self._frame_counter: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._output_dir: str = ""

    def load(self) -> None:
        """初始化后端：创建输出目录，确定起始帧序号。"""
        self._output_dir = self.config.model_path

        # 创建输出目录
        os.makedirs(self._output_dir, exist_ok=True)
        logger.info("BMP后端初始化: 输出目录=%s", self._output_dir)

        # 扫描已有文件，确定起始帧序号（断点续编）
        self._frame_counter = self._scan_existing_frames()
        logger.info("BMP后端就绪: 起始帧序号=%d", self._frame_counter + 1)

    def _scan_existing_frames(self) -> int:
        """扫描目录中已有的帧文件，返回最大帧序号。

        Returns:
            目录中已有的最大帧序号，若无文件则返回0。
        """
        max_index = 0
        pattern = re.compile(r"^frame_(\d{6})\.bmp$")

        try:
            for filename in os.listdir(self._output_dir):
                match = pattern.match(filename)
                if match:
                    index = int(match.group(1))
                    max_index = max(max_index, index)
        except OSError as e:
            logger.warning("扫描目录失败: %s", e)

        return max_index

    def _get_next_frame_path(self) -> str:
        """获取下一帧的文件路径（线程安全）。

        Returns:
            完整的BMP文件路径。
        """
        with self._lock:
            self._frame_counter += 1
            filename = f"frame_{self._frame_counter:06d}.bmp"
            return os.path.join(self._output_dir, filename)

    def analyze(self, image: Image.Image, prompt: str | None = None) -> VLMResult:
        """保存单帧图像为BMP文件。

        Args:
            image: RGB格式的PIL图像。
            prompt: 文本提示词（记录到元信息中，不参与实际处理）。

        Returns:
            VLMResult，text字段为JSON列表格式的文件路径。
        """
        t0 = time.perf_counter()

        # 获取输出路径并保存图像
        file_path = self._get_next_frame_path()
        image.save(file_path, format="BMP")

        elapsed = time.perf_counter() - t0
        image_size = image.size

        logger.debug(
            "BMP保存: %s (%dx%d) %.3fs",
            file_path, image_size[0], image_size[1], elapsed
        )

        # 构建返回结果
        files = [file_path]
        result = VLMResult(
            text=json.dumps({"files": files}),
            inference_time=elapsed,
            meta={
                "backend": "bmp",
                "files": files,
                "frame_index": self._frame_counter,
                "image_size": image_size,
                "prompt": prompt or self.config.default_prompt,
            }
        )

        return result

    def analyze_batch(
        self,
        images: Sequence[Image.Image],
        prompt: str | None = None,
    ) -> VLMResult:
        """批量保存多帧图像为BMP文件。

        Args:
            images: RGB格式的PIL图像序列。
            prompt: 文本提示词（记录到元信息中）。

        Returns:
            VLMResult，text字段为JSON列表格式的所有文件路径。
        """
        if not images:
            return VLMResult(
                text=json.dumps({"files": []}),
                inference_time=0.0,
                meta={"backend": "bmp", "files": [], "frame_indices": []}
            )

        t0 = time.perf_counter()

        files: List[str] = []
        frame_indices: List[int] = []
        image_sizes: List[tuple] = []

        for img in images:
            file_path = self._get_next_frame_path()
            img.save(file_path, format="BMP")

            files.append(file_path)
            frame_indices.append(self._frame_counter)
            image_sizes.append(img.size)

        elapsed = time.perf_counter() - t0

        logger.debug(
            "BMP批量保存: %d帧 -> %s (%.3fs)",
            len(images), self._output_dir, elapsed
        )

        return VLMResult(
            text=json.dumps({"files": files}),
            inference_time=elapsed,
            meta={
                "backend": "bmp",
                "files": files,
                "frame_indices": frame_indices,
                "image_sizes": image_sizes,
                "prompt": prompt or self.config.default_prompt,
            }
        )

    def unload(self) -> None:
        """清理资源，记录统计信息。"""
        logger.info(
            "BMP后端卸载: 共保存 %d 帧 -> %s",
            self._frame_counter, self._output_dir
        )
        self._frame_counter = 0
        self._output_dir = ""

    def _get_default_prompt(self) -> str:
        """获取默认提示词。"""
        return self.config.default_prompt

    def supports_batch(self) -> bool:
        """BMP后端支持批量处理。"""
        return True
