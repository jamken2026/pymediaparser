"""文件读取器：从本地/网络视频文件或图片文件中逐帧读取。

基于 PyAV（FFmpeg）实现视频解码，基于 PIL 实现图片加载。
输出格式与 StreamReader 完全一致，可直接对接 FrameSampler。

典型用法::

    from pymediaparser.vlm_base import StreamConfig
    from pymediaparser.file_reader import FileReader

    cfg = StreamConfig(url="/path/to/video.mp4")
    with FileReader(cfg) as reader:
        for frame, ts in reader.frames():
            # frame: av.VideoFrame, ts: float (秒)
            ...

    # 图片文件
    cfg = StreamConfig(url="/path/to/image.jpg")
    with FileReader(cfg) as reader:
        for frame, ts in reader.frames():
            # 单帧输出
            ...
"""

from __future__ import annotations

import io
import logging
import urllib.request
from typing import Iterator
from urllib.parse import urlparse

import av
import numpy as np
from PIL import Image

from .vlm_base import StreamConfig

logger = logging.getLogger(__name__)

# 支持的图片扩展名
_IMAGE_EXTENSIONS = frozenset({
    ".bmp", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".tif",
})


class FileReader:
    """从文件（本地路径或网络 URL）读取视频帧或图片。

    支持的视频格式：mp4, mov, m3u8, ts, ps, flv, avi, mkv 等（FFmpeg 支持的所有格式）。
    支持的图片格式：bmp, jpg, jpeg, png, gif, webp, tiff 等。

    输出接口与 ``StreamReader`` 保持一致：``frames()`` 生成器 yield
    ``(av.VideoFrame, timestamp_seconds)`` 元组，可直接传入 ``FrameSampler``。

    Args:
        config: 流配置。``url`` 字段为文件路径或网络 URL。
    """

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._container: av.InputContainer | None = None
        self._is_image: bool = False
        self._total_frames: int = 0
        self._duration_seconds: float = 0.0
        self._pil_image: Image.Image | None = None  # 图片模式缓存
        self._closed: bool = False  # 关闭标志

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """视频总帧数（0 表示未知或未打开），图片固定为 1。"""
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        """视频总时长（秒），0.0 表示未知或未打开。图片固定为 0.0。"""
        return self._duration_seconds

    @property
    def is_image(self) -> bool:
        """当前文件是否为图片。"""
        return self._is_image

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def open(self) -> None:
        """打开文件并收集元信息。

        Raises:
            FileNotFoundError: 本地文件不存在。
            av.AVError: 视频文件打开或探测失败。
            Exception: 图片文件加载失败。
        """
        self._is_image = self._detect_is_image()

        if self._is_image:
            self._open_image()
        else:
            self._open_video()

    def frames(self) -> Iterator[tuple[av.VideoFrame, float]]:
        """逐帧迭代文件内容。

        Yields:
            ``(video_frame, timestamp_seconds)`` 元组。

        视频文件：按解码顺序逐帧输出，文件结束后自然退出。
        图片文件：输出单帧后结束。
        """
        if self._is_image:
            yield from self._image_frames()
        else:
            yield from self._video_frames()

    def close(self) -> None:
        """关闭文件并释放资源。"""
        self._closed = True  # 先设置标志，让迭代器退出
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
        self._pil_image = None
        logger.info("文件读取器已关闭")

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> "FileReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 内部方法 —— 类型检测
    # ------------------------------------------------------------------

    def _detect_is_image(self) -> bool:
        """根据 URL/路径的扩展名判断是否为图片文件。"""
        url = self.config.url
        parsed = urlparse(url)
        # 对本地路径和网络 URL 统一取 path 部分
        path = parsed.path if parsed.scheme else url
        # 取最后一段的扩展名
        dot_idx = path.rfind(".")
        if dot_idx == -1:
            return False
        ext = path[dot_idx:].lower()
        return ext in _IMAGE_EXTENSIONS

    # ------------------------------------------------------------------
    # 内部方法 —— 图片处理
    # ------------------------------------------------------------------

    def _open_image(self) -> None:
        """打开图片文件（本地或网络）。"""
        url = self.config.url
        if self._is_network_url(url):
            logger.info("正在下载图片: %s", url)
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            self._pil_image = Image.open(io.BytesIO(data))
        else:
            logger.info("正在打开图片: %s", url)
            self._pil_image = Image.open(url)

        # 确保图片数据被完全加载（避免延迟 IO 问题）
        self._pil_image.load()
        self._total_frames = 1
        self._duration_seconds = 0.0
        logger.info("图片已加载: %s (%dx%d)", url, self._pil_image.width, self._pil_image.height)

    def _image_frames(self) -> Iterator[tuple[av.VideoFrame, float]]:
        """将图片作为单帧 av.VideoFrame 输出。"""
        if self._pil_image is None:
            return

        # PIL → RGB numpy → av.VideoFrame
        img = self._pil_image.convert("RGB")
        arr = np.array(img)
        video_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        yield video_frame, 0.0

    # ------------------------------------------------------------------
    # 内部方法 —— 视频处理
    # ------------------------------------------------------------------

    def _open_video(self) -> None:
        """打开视频文件（本地或网络 URL）。"""
        url = self.config.url
        options = self._build_options()

        logger.info("正在打开视频文件: %s", url)

        kwargs: dict = {"options": options}
        # 对网络 URL 设置超时
        if self._is_network_url(url):
            timeout_us = int(self.config.timeout * 1_000_000)
            kwargs["options"].setdefault("rw_timeout", str(timeout_us))

        self._container = av.open(url, **kwargs)

        # 收集元信息
        video_stream = self._container.streams.video[0]
        video_stream.thread_type = "AUTO"

        # 设置解码模式
        if self.config.decode_mode == "keyframe_only":
            video_stream.codec_context.skip_frame = "NONKEY"
            logger.info("解码模式: 仅关键帧(I帧)")

        # 总帧数（某些容器格式可能返回 0）
        self._total_frames = video_stream.frames or 0

        # 总时长（container.duration 单位为微秒，可能为 None）
        if self._container.duration is not None:
            self._duration_seconds = self._container.duration / 1_000_000.0
        else:
            self._duration_seconds = 0.0

        logger.info(
            "视频文件已打开: %s (总帧数=%s, 时长=%.1fs)",
            url,
            self._total_frames if self._total_frames > 0 else "未知",
            self._duration_seconds,
        )

    def _video_frames(self) -> Iterator[tuple[av.VideoFrame, float]]:
        """逐帧解码视频文件。"""
        if self._container is None:
            return

        try:
            for frame in self._container.decode(video=0):
                # 检查关闭标志，提前退出
                if self._closed:
                    return
                ts = self._frame_timestamp(frame)
                yield frame, ts
                # yield 后再次检查，确保及时响应关闭请求
                if self._closed:
                    return
        except GeneratorExit:
            return
        except av.AVError as exc:
            # 容器被关闭时会抛出此异常，正常退出
            logger.debug("视频解码中断（容器已关闭）: %s", exc)
        except Exception as exc:
            # 其他异常只记录日志，不抛出（避免线程崩溃）
            if self._container is not None and not self._closed:
                logger.error("视频文件读取错误: %s", exc, exc_info=True)

    def _build_options(self) -> dict[str, str]:
        """构建 FFmpeg 选项（文件模式，不需要低延迟参数）。"""
        return {}

    @staticmethod
    def _frame_timestamp(frame: av.VideoFrame) -> float:
        """从帧中提取时间戳（秒）。"""
        if frame.pts is not None and frame.time_base is not None:
            return float(frame.pts * frame.time_base)
        if frame.time is not None:
            return float(frame.time)
        return 0.0

    @staticmethod
    def _is_network_url(url: str) -> bool:
        """判断是否为网络 URL。"""
        return url.lower().startswith(("http://", "https://"))
