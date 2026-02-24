#!/usr/bin/env python3
"""智能Pipeline完整使用示例"""

import logging
import time
from src.python_test import (
    SmartLivePipeline,
    StreamConfig, 
    VLMConfig,
    Qwen2VLClient,
    ConsoleResultHandler
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def demo_traditional_vs_smart():
    """传统模式 vs 智能模式对比演示"""
    print("=" * 60)
    print("🤖 智能Pipeline功能演示")
    print("=" * 60)
    
    # 基础配置
    stream_url = "rtmp://your-actual-stream-address/live/stream"  # 请替换为实际流地址
    stream_config = StreamConfig(url=stream_url, target_fps=1.0, max_queue_size=5)
    vlm_config = VLMConfig(device="cuda:0", max_new_tokens=256)
    vlm_client = Qwen2VLClient(vlm_config)
    
    print("演示配置:")
    print(f"  流地址: {stream_config.url}")
    print(f"  抽帧频率: {stream_config.target_fps} fps")
    print(f"  推理设备: {vlm_config.device}")
    print()
    
    # 1. 传统模式演示
    print("1️⃣ 传统模式 (固定频率采样)")
    print("-" * 40)
    traditional_pipeline = SmartLivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        handlers=[ConsoleResultHandler()],
        enable_smart_sampling=False,
        enable_batch_processing=False
    )
    print("✅ 配置完成 - 处理所有帧，无优化")
    
    # 2. 智能采样模式演示
    print("\n2️⃣ 智能采样模式")
    print("-" * 40)
    smart_pipeline = SmartLivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        handlers=[ConsoleResultHandler()],
        enable_smart_sampling=True,
        enable_batch_processing=False,
        motion_method="MOG2",
        ssim_threshold=0.85
    )
    print("✅ 配置完成 - 基于内容变化智能采样")
    print(f"   运动检测: {smart_pipeline.smart_sampler.motion_detector.method}")
    print(f"   变化阈值: {smart_pipeline.smart_sampler.change_analyzer.ssim_threshold}")
    
    # 3. 完整智能模式演示
    print("\n3️⃣ 完整智能模式")
    print("-" * 40)
    full_smart_pipeline = SmartLivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        handlers=[ConsoleResultHandler()],
        enable_smart_sampling=True,
        enable_batch_processing=True,
        motion_method="MOG2",
        ssim_threshold=0.85,
        batch_buffer_size=5,
        batch_timeout=2.0
    )
    print("✅ 配置完成 - 智能采样 + 批量处理")
    print(f"   批缓冲区: {full_smart_pipeline.batch_processor.frame_buffer.max_size}")
    print(f"   批超时: {full_smart_pipeline.batch_processor.frame_buffer.max_wait_time}秒")
    
    print("\n" + "=" * 60)
    print("🎯 性能优化预期:")
    print("=" * 60)
    print("传统模式    : 100% 帧处理, 100% VLM调用 (基准)")
    print("智能采样    : 30-70% 帧处理, 30-70% VLM调用 (降低30-70%)")
    print("批量处理    : 100% 帧处理, 30-50% VLM调用 (降低50-70%)") 
    print("完整智能    : 30-70% 帧处理, 10-30% VLM调用 (降低70-90%)")
    print("=" * 60)
    
    print("\n💡 使用说明:")
    print("1. 请将上面的流地址替换为真实的RTMP/HTTP流地址")
    print("2. 首次使用建议用CPU模式测试基础功能")
    print("3. 生产环境推荐使用GPU模式获得最佳性能")
    print("4. 可通过 --smart-sampling 和 --batch-processing 参数控制功能开关")
    
    # 实际运行示例（注释掉避免意外执行）
    # print("\n🚀 开始运行完整智能模式演示...")
    # try:
    #     full_smart_pipeline.run()
    # except KeyboardInterrupt:
    #     print("演示已停止")

if __name__ == "__main__":
    demo_traditional_vs_smart()
