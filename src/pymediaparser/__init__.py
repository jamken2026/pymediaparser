"""pymediaparser - 人物检测与视觉语言模型工具包

包含：
- vlm_base: VLM 配置/结果数据类与抽象接口
- vlm_qwen2: Qwen2-VL 视觉语言模型推理
- stream_reader: RTMP/HTTP-FLV/HTTP-TS 实时流接入
- frame_sampler: 按频率抽帧
- result_handler: 结果输出处理器
- live_pipeline: 实时流 VLM 分析 Pipeline

使用示例::

    # 方式 1：CLI 命令行直接运行
    # python -m pymediaparser.live_pipeline --url rtmp://host/live/stream --fps 1

    # 方式 2：Python API
    from pymediaparser import LivePipeline, StreamConfig, VLMConfig, Qwen2VLClient

    stream_cfg = StreamConfig(url="rtmp://host/live/stream", target_fps=1.0)
    vlm_client = Qwen2VLClient(VLMConfig(device="cuda:0"))
    pipeline = LivePipeline(stream_cfg, vlm_client)
    pipeline.run()
"""

# 延迟导入：避免 python -m pymediaparser.live_pipeline 时的 RuntimeWarning
# 用户通过 from pymediaparser import XXX 时会触发 __getattr__ 按需加载。

__all__ = [
    # 配置与数据类
    'StreamConfig',
    'VLMConfig',
    'VLMResult',
    'FrameResult',
    # VLM 客户端（Qwen2VLClient已集成批量处理功能）
    'VLMClient',
    'Qwen2VLClient',
    # 流与采样
    'StreamReader',
    'FrameSampler',
    'SmartSampler',
    'SimpleSmartSampler',
    'MLSmartSampler',
    'MotionDetector',
    'ChangeAnalyzer',
    'ForegroundExtractor',
    # 帧缓存
    'FrameBuffer',
    # 结果处理
    'ResultHandler',
    'ConsoleResultHandler',
    'HttpCallbackHandler',
    # Pipeline（统一入口，支持传统/智能双模式）
    'LivePipeline',
]


def __getattr__(name: str):
    """延迟导入：按需加载子模块符号。"""
    _import_map = {
        'StreamConfig':          ('.vlm_base', 'StreamConfig'),
        'VLMConfig':             ('.vlm_base', 'VLMConfig'),
        'VLMResult':             ('.vlm_base', 'VLMResult'),
        'FrameResult':           ('.vlm_base', 'FrameResult'),
        'VLMClient':             ('.vlm_base', 'VLMClient'),
        'StreamReader':          ('.stream_reader', 'StreamReader'),
        'FrameSampler':          ('.frame_sampler', 'FrameSampler'),
        'Qwen2VLClient':         ('.vlm_qwen2', 'Qwen2VLClient'),
        'ResultHandler':         ('.result_handler', 'ResultHandler'),
        'ConsoleResultHandler':  ('.result_handler', 'ConsoleResultHandler'),
        'HttpCallbackHandler':   ('.result_handler', 'HttpCallbackHandler'),
        'LivePipeline':          ('.live_pipeline', 'LivePipeline'),
        # 新增智能采样相关
        'SmartSampler':          ('.smart_sampler.base', 'SmartSampler'),
        'SimpleSmartSampler':    ('.smart_sampler.simple_smart_sampler', 'SimpleSmartSampler'),
        'MLSmartSampler':        ('.smart_sampler.ml_smart_sampler', 'MLSmartSampler'),
        'MotionDetector':        ('.smart_sampler.motion_detector', 'MotionDetector'),
        'ChangeAnalyzer':        ('.smart_sampler.change_analyzer', 'ChangeAnalyzer'),
        'ForegroundExtractor':   ('.smart_sampler.foreground_extractor', 'ForegroundExtractor'),
        # 帧缓存
        'FrameBuffer':           ('.frame_buffer', 'FrameBuffer'),
    }
    if name in _import_map:
        module_path, attr = _import_map[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
