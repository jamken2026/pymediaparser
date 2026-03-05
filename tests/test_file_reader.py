"""FileReader 单元测试。

使用项目中已有的测试资源文件：
- resource/866f893bb7a353c75f1aa5c7cb61e4e3.mp4 (视频)
- resource/test_img1.png (图片)
"""

from __future__ import annotations

import os
import pytest

import av

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOURCE_DIR = os.path.join(_PROJECT_ROOT, "resource")

# 测试资源文件
_VIDEO_FILE = os.path.join(_RESOURCE_DIR, "866f893bb7a353c75f1aa5c7cb61e4e3.mp4")
_IMAGE_FILE = os.path.join(_RESOURCE_DIR, "test_img1.png")


# =====================================================================
# 跳过条件：如果资源文件不存在
# =====================================================================

skip_if_no_video = pytest.mark.skipif(
    not os.path.exists(_VIDEO_FILE),
    reason=f"测试视频文件不存在: {_VIDEO_FILE}",
)

skip_if_no_image = pytest.mark.skipif(
    not os.path.exists(_IMAGE_FILE),
    reason=f"测试图片文件不存在: {_IMAGE_FILE}",
)


# =====================================================================
# FileReader 测试
# =====================================================================

class TestFileReaderTypeDetection:
    """文件类型检测测试。"""

    def test_detect_video_mp4(self):
        """mp4 文件应识别为视频。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/path/to/video.mp4")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is False

    def test_detect_video_mov(self):
        """mov 文件应识别为视频。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/path/to/video.mov")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is False

    def test_detect_video_ts(self):
        """ts 文件应识别为视频。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/path/to/video.ts")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is False

    def test_detect_image_png(self):
        """png 文件应识别为图片。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/path/to/image.png")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is True

    def test_detect_image_jpg(self):
        """jpg 文件应识别为图片。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/path/to/image.jpg")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is True

    def test_detect_image_jpeg(self):
        """jpeg 文件应识别为图片。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/path/to/image.jpeg")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is True

    def test_detect_image_bmp(self):
        """bmp 文件应识别为图片。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/path/to/image.bmp")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is True

    def test_detect_url_video(self):
        """网络视频 URL 应识别为视频。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="http://example.com/video.mp4")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is False

    def test_detect_url_image(self):
        """网络图片 URL 应识别为图片。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="http://example.com/image.png")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is True

    def test_detect_no_extension(self):
        """无扩展名默认视频。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/path/to/file")
        reader = FileReader(cfg)
        assert reader._detect_is_image() is False


@skip_if_no_video
class TestFileReaderVideo:
    """视频文件读取测试。"""

    def test_open_video(self):
        """打开视频文件成功。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url=_VIDEO_FILE)
        reader = FileReader(cfg)
        reader.open()

        assert reader.is_image is False
        assert reader.total_frames >= 0  # 可能为 0（未知）
        assert reader.duration_seconds >= 0.0

        reader.close()

    def test_video_frames_output_format(self):
        """视频帧输出格式正确：(av.VideoFrame, float)。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url=_VIDEO_FILE)
        with FileReader(cfg) as reader:
            frame_count = 0
            for frame, ts in reader.frames():
                assert isinstance(frame, av.VideoFrame)
                assert isinstance(ts, float)
                assert ts >= 0.0
                frame_count += 1
                if frame_count >= 5:  # 只测试前 5 帧
                    break

            assert frame_count > 0

    def test_video_all_frames(self):
        """视频所有帧都能读取。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url=_VIDEO_FILE)
        with FileReader(cfg) as reader:
            frames = list(reader.frames())
            assert len(frames) > 0
            # 每帧都应该是有效的元组
            for frame, ts in frames:
                assert isinstance(frame, av.VideoFrame)
                assert isinstance(ts, float)

    def test_video_context_manager(self):
        """上下文管理器正常工作。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url=_VIDEO_FILE)
        with FileReader(cfg) as reader:
            assert reader.is_image is False
        # 退出后资源应释放


@skip_if_no_image
class TestFileReaderImage:
    """图片文件读取测试。"""

    def test_open_image(self):
        """打开图片文件成功。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url=_IMAGE_FILE)
        reader = FileReader(cfg)
        reader.open()

        assert reader.is_image is True
        assert reader.total_frames == 1
        assert reader.duration_seconds == 0.0

        reader.close()

    def test_image_single_frame(self):
        """图片只输出单帧。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url=_IMAGE_FILE)
        with FileReader(cfg) as reader:
            frames = list(reader.frames())
            assert len(frames) == 1

    def test_image_frame_format(self):
        """图片帧输出格式正确：(av.VideoFrame, 0.0)。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url=_IMAGE_FILE)
        with FileReader(cfg) as reader:
            for frame, ts in reader.frames():
                assert isinstance(frame, av.VideoFrame)
                assert ts == 0.0

    def test_image_context_manager(self):
        """上下文管理器正常工作。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url=_IMAGE_FILE)
        with FileReader(cfg) as reader:
            assert reader.is_image is True


class TestFileReaderErrors:
    """错误处理测试。"""

    def test_file_not_found(self):
        """不存在的文件应抛出异常。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.vlm_base import StreamConfig

        cfg = StreamConfig(url="/nonexistent/path/video.mp4")
        reader = FileReader(cfg)

        with pytest.raises(Exception):  # FileNotFoundError 或 av.AVError
            reader.open()


@skip_if_no_video
@skip_if_no_image
class TestFileReaderWithFrameSampler:
    """FileReader 与 FrameSampler 集成测试。"""

    def test_video_with_sampler(self):
        """视频帧能通过 FrameSampler 正确采样。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.frame_sampler import FrameSampler
        from pymediaparser.vlm_base import StreamConfig
        from PIL import Image

        cfg = StreamConfig(url=_VIDEO_FILE, target_fps=1.0)
        sampler = FrameSampler(target_fps=cfg.target_fps)

        with FileReader(cfg) as reader:
            sampled = list(sampler.sample(reader.frames()))
            assert len(sampled) > 0

            for image, ts, idx in sampled:
                assert isinstance(image, Image.Image)
                assert isinstance(ts, float)
                assert isinstance(idx, int)

    def test_image_with_sampler(self):
        """图片帧能通过 FrameSampler 正确采样。"""
        from pymediaparser.file_reader import FileReader
        from pymediaparser.frame_sampler import FrameSampler
        from pymediaparser.vlm_base import StreamConfig
        from PIL import Image

        cfg = StreamConfig(url=_IMAGE_FILE, target_fps=1.0)
        sampler = FrameSampler(target_fps=cfg.target_fps)

        with FileReader(cfg) as reader:
            sampled = list(sampler.sample(reader.frames()))
            assert len(sampled) == 1  # 图片只有一帧

            image, ts, idx = sampled[0]
            assert isinstance(image, Image.Image)
            assert ts == 0.0
            assert idx == 0
