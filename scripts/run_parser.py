#!/usr/bin/env python3
"""媒体流解析启动脚本。

从 RTMP / HTTP-FLV / HTTP-TS 实时流中按指定频率抽帧，
送入 Qwen2-VL 大模型进行画面理解，并将结果输出到控制台。

用法示例::

    # RTMP 流，每秒 1 帧，使用 GPU 推理
    python scripts/run_parser.py --url rtmp://host/live/stream

    # HTTP-FLV 流，每 2 秒 1 帧，自定义 prompt
    python scripts/run_parser.py \\
        --url http://host/live/stream.flv \\
        --fps 0.5 \\
        --prompt "识别画面中的人物并描述他们的行为。"

    # HTTP-TS 流，指定模型路径
    python scripts/run_parser.py \\
        --url http://host/live/stream.ts \\
        --model-path /path/to/Qwen2-VL-2B-Instruct

    # CPU 模式（无 GPU 时）
    python scripts/run_parser.py \\
        --url rtmp://host/live/stream \\
        --device cpu --dtype float32 --fps 0.2
"""

import argparse
import logging
import os
import sys

# 确保项目 src 目录在搜索路径中（支持直接 python scripts/run_parser.py 运行）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pymediaparser.vlm_base import StreamConfig, VLMConfig
from pymediaparser.vlm_qwen2 import Qwen2VLClient
from pymediaparser.result_handler import ConsoleResultHandler, HttpCallbackHandler
from pymediaparser.live_pipeline import LivePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实时流 VLM 分析 —— 拉取视频流、抽帧、Qwen2-VL 推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  %(prog)s --url rtmp://192.168.1.100/live/stream
  %(prog)s --url http://host/live/stream.flv --fps 0.5
  %(prog)s --url http://host/live/stream.ts --prompt "描述画面中的人物活动"
  
智能采样模式:
  %(prog)s --url rtmp://host/live/stream --smart-sampling
  %(prog)s --url rtmp://host/live/stream --smart-sampling --motion-threshold 0.2
  
批量处理模式（可独立使用或配合智能采样）:
  %(prog)s --url rtmp://host/live/stream --batch-processing
  %(prog)s --url rtmp://host/live/stream --smart-sampling --batch-processing
""",
    )

    # ── 流配置 ──────────────────────────────────────────────
    stream_group = parser.add_argument_group("流配置")
    stream_group.add_argument(
        "--url", required=True,
        help="视频流地址 (rtmp:// / http://*.flv / http://*.ts / *.m3u8)",
    )
    stream_group.add_argument(
        "--format", default=None,
        choices=["flv", "mpegts"],
        help="强制指定容器格式（默认根据 URL 自动检测）",
    )
    stream_group.add_argument(
        "--fps", type=float, default=1.0,
        help="目标抽帧频率，帧/秒（默认: 1.0）",
    )
    stream_group.add_argument(
        "--queue-size", type=int, default=3,
        help="帧缓冲队列最大长度，超出时丢弃旧帧（默认: 3）",
    )
    stream_group.add_argument(
        "--reconnect", type=float, default=3.0,
        help="断线重连等待秒数（默认: 3.0）",
    )
    stream_group.add_argument(
        "--timeout", type=float, default=10.0,
        help="流读取超时秒数（默认: 10.0）",
    )

    # ── VLM 配置 ─────────────────────────────────────────────
    vlm_group = parser.add_argument_group("VLM 模型配置")
    vlm_group.add_argument(
        "--model-path", default=None,
        help="Qwen2-VL 模型本地路径（默认: 项目内置 models/Qwen/Qwen2-VL-2B-Instruct）",
    )
    vlm_group.add_argument(
        "--device", default="cuda:0",
        help="推理设备（默认: cuda:0）",
    )
    vlm_group.add_argument(
        "--dtype", default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="推理精度（默认: float16）",
    )
    vlm_group.add_argument(
        "--max-tokens", type=int, default=256,
        help="单次生成最大 token 数（默认: 256）",
    )
    vlm_group.add_argument(
        "--prompt", default=None,
        help="VLM 提示词（默认: '请描述当前画面中的人物活动。'）",
    )
    vlm_group.add_argument(
        "--no-flash-attn", action="store_true",
        help="禁用 Flash Attention 2",
    )
    
    # ── 智能抽帧配置 ─────────────────────────────────────────
    smart_group = parser.add_argument_group("智能抽帧配置")
    smart_group.add_argument(
        "--smart-sampling", action="store_true",
        help="启用智能抽帧模式（基于内容变化检测）",
    )
    smart_group.add_argument(
        "--motion-method", default="MOG2",
        choices=["MOG2", "KNN"],
        help="运动检测方法（默认: MOG2）",
    )
    smart_group.add_argument(
        "--motion-threshold", type=float, default=0.1,
        help="运动检测阈值，运动像素占比（默认: 0.1）",
    )
    smart_group.add_argument(
        "--backup-interval", type=float, default=30.0,
        help="保底采样间隔（秒），画面无变化时的最大间隔（默认: 30）",
    )
    smart_group.add_argument(
        "--min-frame-interval", type=float, default=1.0,
        help="最小帧间隔（秒），避免连续送帧导致VLM过载（默认: 1）",
    )
    
    # ── 批量处理配置 ─────────────────────────────────────────
    batch_group = parser.add_argument_group("批量处理配置（可独立使用）")
    batch_group.add_argument(
        "--batch-processing", action="store_true",
        help="启用批量处理模式：积累多帧后批量送入VLM理解",
    )
    batch_group.add_argument(
        "--batch-buffer-size", type=int, default=5,
        help="批量处理缓冲区大小（默认: 5）",
    )
    batch_group.add_argument(
        "--batch-timeout", type=float, default=2.0,
        help="批量处理超时时间（秒，默认: 2.0）",
    )

    # ── 输出配置 ─────────────────────────────────────────────
    output_group = parser.add_argument_group("输出配置")
    output_group.add_argument(
        "--callback-url", default=None,
        help="HTTP 回调地址（可选），设置后将同时通过 POST 推送结果",
    )
    output_group.add_argument(
        "--quiet", action="store_true",
        help="静默模式：不在控制台打印分析结果（仅配合 --callback-url 使用）",
    )

    # ── 日志 ─────────────────────────────────────────────────
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 配置日志 ──────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run_live_vlm")

    # ── 构建 StreamConfig ─────────────────────────────────────
    stream_cfg = StreamConfig(
        url=args.url,
        format=args.format,
        target_fps=args.fps,
        reconnect_interval=args.reconnect,
        timeout=args.timeout,
        max_queue_size=args.queue_size,
    )

    # ── 构建 VLMConfig ───────────────────────────────────────
    vlm_kwargs = {
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_tokens,
        "use_flash_attn": not args.no_flash_attn,
    }
    if args.model_path:
        vlm_kwargs["model_path"] = args.model_path
    if args.prompt:
        vlm_kwargs["default_prompt"] = args.prompt

    vlm_cfg = VLMConfig(**vlm_kwargs)
    vlm_client = Qwen2VLClient(vlm_cfg)

    # ── 构建 ResultHandler 列表 ───────────────────────────────
    handlers = []
    if not args.quiet:
        handlers.append(ConsoleResultHandler(verbose=True))
    if args.callback_url:
        handlers.append(HttpCallbackHandler(callback_url=args.callback_url))
    if not handlers:
        # quiet 模式下也没有 callback，至少保留控制台输出
        logger.warning("未配置任何输出通道，将默认使用控制台输出")
        handlers.append(ConsoleResultHandler(verbose=True))

    # ── 打印启动信息 ──────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("实时流 VLM 分析")
    logger.info("=" * 50)
    logger.info("流地址:     %s", stream_cfg.url)
    logger.info("抽帧频率:   %.2f fps", stream_cfg.target_fps)
    logger.info("推理设备:   %s", vlm_cfg.device)
    logger.info("推理精度:   %s", vlm_cfg.dtype)
    logger.info("最大 tokens: %d", vlm_cfg.max_new_tokens)
    logger.info("提示词:     %s", vlm_cfg.default_prompt)
    logger.info("队列大小:   %d", stream_cfg.max_queue_size)
    if args.callback_url:
        logger.info("HTTP 回调:  %s", args.callback_url)
    logger.info("=" * 50)

    # ── 构建 Pipeline ────────────────────────────────────────
    # 统一使用 LivePipeline，通过参数控制模式
    smart_config = None
    if args.smart_sampling or args.batch_processing:
        smart_config = {
            'motion_method': args.motion_method,
            'motion_threshold': args.motion_threshold,
            'backup_interval': args.backup_interval,
            'min_frame_interval': args.min_frame_interval,
            'batch_buffer_size': args.batch_buffer_size,
            'batch_timeout': args.batch_timeout,
        }
    
    pipeline = LivePipeline(
        stream_config=stream_cfg,
        vlm_client=vlm_client,
        handlers=handlers,
        prompt=args.prompt,
        enable_smart_sampling=args.smart_sampling,
        enable_batch_processing=args.batch_processing,
        smart_config=smart_config,
    )
    
    # 显示运行模式信息
    if args.smart_sampling or args.batch_processing:
        mode_info = []
        if args.smart_sampling:
            mode_info.append("智能采样")
        if args.batch_processing:
            mode_info.append("批量处理")
        logger.info("运行模式:   %s", "+".join(mode_info))
        
        if args.smart_sampling:
            logger.info("运动检测:   %s", args.motion_method)
            logger.info("运动阈值:   %.2f", args.motion_threshold)
            logger.info("保底间隔:   %.1f秒", args.backup_interval)
            logger.info("最小帧间隔: %.1f秒", args.min_frame_interval)
        if args.batch_processing:
            logger.info("批缓冲区:   %d", args.batch_buffer_size)
            logger.info("批超时:     %.1f秒", args.batch_timeout)
    else:
        logger.info("运行模式:   传统固定频率抽帧")

    logger.info("=" * 50)
    
    # ── 运行 Pipeline ────────────────────────────────────────
    pipeline.run()


if __name__ == "__main__":
    main()
