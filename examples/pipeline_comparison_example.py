#!/usr/bin/env python3
"""LivePipeline vs ReplayPipeline 对比示例

展示两个 Pipeline 类的核心差异和适用场景，帮助用户选择合适的 Pipeline 类型。

VLM 后端选择：
- bmp: 虚拟调试后端，保存图像为 BMP 文件（无需 GPU，推荐用于入门和调试）
- qwen35: Qwen3.5 本地模型推理（需要 GPU 和模型文件）
- openai_api: OpenAI 兼容 API 服务（需要 API 服务地址）
"""

import logging
import tempfile
from pymediaparser import (
    LivePipeline,
    ReplayPipeline,
    StreamConfig,
    VLMConfig,
    create_vlm_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def show_comparison():
    """展示两个 Pipeline 的核心差异"""
    
    print("\n" + "=" * 70)
    print("LivePipeline vs ReplayPipeline 对比")
    print("=" * 70)
    
    comparison = """
┌─────────────────┬──────────────────────────┬──────────────────────────┐
│     特性        │      LivePipeline        │      ReplayPipeline      │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ 输入源          │ 实时流 (RTMP/HTTP-FLV)   │ 文件 (视频/图片)         │
│ 退出方式        │ 手动停止 (Ctrl+C)        │ 自动退出 (文件处理完毕)  │
│ 队列策略        │ 丢弃旧帧（保证实时性）   │ 阻塞等待（保证不丢帧）   │
│ 进度跟踪        │ 无总进度                 │ 有总进度和百分比         │
│ 完成状态        │ STOPPED (手动停止)       │ COMPLETED (正常完成)     │
│ 适用场景        │ 实时监控、直播分析       │ 离线分析、批量处理       │
│ 重连机制        │ 支持断线重连             │ 不适用                   │
│ 内存管理        │ 严格限制（队列满丢帧）   │ 相对宽松（阻塞等待）     │
└─────────────────┴──────────────────────────┴──────────────────────────┘
    """
    print(comparison)


def show_use_cases():
    """展示适用场景"""
    
    print("\n" + "=" * 70)
    print("适用场景")
    print("=" * 70)
    
    print("""
【LivePipeline 适用场景】

1. 实时监控分析
   - 安防监控场景理解
   - 直播间内容审核
   - 工业产线异常检测

2. 直播流处理
   - 直播内容实时摘要
   - 弹幕互动分析
   - 主播行为分析

3. 持续运行的服务
   - 7x24小时监控服务
   - 边缘设备实时推理
   - 云端流处理服务

【ReplayPipeline 适用场景】

1. 离线视频分析
   - 历史录像内容理解
   - 视频档案索引生成
   - 批量视频处理任务

2. 单张图片分析
   - 图片内容识别
   - 批量图片标注
   - 图片审核服务

3. 一次性任务
   - 视频内容摘要
   - 精彩片段提取
   - 视频质量评估
    """)


def show_code_comparison():
    """展示代码层面的对比"""
    
    print("\n" + "=" * 70)
    print("代码对比")
    print("=" * 70)
    
    print("""
【LivePipeline 基础用法（BMP 后端 - 推荐入门）】

    from pymediaparser import LivePipeline, StreamConfig, create_vlm_client, VLMConfig
    import tempfile
    
    # 配置实时流
    stream_cfg = StreamConfig(
        url="rtmp://host/live/stream",
        target_fps=1.0,
    )
    
    # 创建 BMP 后端客户端（无需 GPU）
    output_dir = tempfile.mkdtemp(prefix="live_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    # 创建并运行 Pipeline
    pipeline = LivePipeline(stream_cfg, vlm_client)
    pipeline.run()  # 阻塞运行，Ctrl+C 停止
    # 帧图像将保存到 output_dir 目录

【ReplayPipeline 基础用法（BMP 后端 - 推荐入门）】

    from pymediaparser import ReplayPipeline, StreamConfig, create_vlm_client, VLMConfig
    import tempfile
    
    # 配置文件路径
    stream_cfg = StreamConfig(
        url="/path/to/video.mp4",  # 本地文件路径
        target_fps=1.0,
    )
    
    # 创建 BMP 后端客户端（无需 GPU）
    output_dir = tempfile.mkdtemp(prefix="replay_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    # 创建并运行 Pipeline
    pipeline = ReplayPipeline(stream_cfg, vlm_client)
    pipeline.run()  # 阻塞运行，处理完自动退出
    # 帧图像将保存到 output_dir 目录

【Qwen3.5 本地模型用法】

    # 安装依赖: pip install 'pymediaparser[vlm-qwen]'
    vlm_config = VLMConfig(
        model_path="/path/to/Qwen3.5-0.8B",
        device="cuda:0",
        dtype="float16",
    )
    vlm_client = create_vlm_client("qwen35", vlm_config)
    # ... 其余配置相同

【OpenAI API 后端用法】

    from pymediaparser import APIVLMConfig
    
    api_config = APIVLMConfig(
        base_url="http://localhost:8000/v1",
        model_name="Qwen2-VL-7B-Instruct",
    )
    vlm_client = create_vlm_client("openai_api", api_config)
    # ... 其余配置相同

【共同点】

两者共享相同的:
- StreamConfig 配置类
- VLMConfig / APIVLMConfig 配置类
- create_vlm_client 工厂函数
- 智能采样和批量处理参数
- 结果处理器接口

【差异点】

LivePipeline 特有:
- reconnect_interval: 断线重连间隔
- 队列满时丢弃旧帧策略

ReplayPipeline 特有:
- 自动检测文件总帧数和时长
- 处理完成后进入 COMPLETED 状态
- 阻塞式队列（不丢帧）
    """)


def show_state_machine():
    """展示状态机对比"""
    
    print("\n" + "=" * 70)
    print("状态机对比")
    print("=" * 70)
    
    print("""
【LivePipeline 状态流转】

    IDLE ──start()──> STARTING ──> RUNNING <──────┐
                       │                          │
                       │    ┌─────────────────────┤
                       │    │                     │
                       ▼    ▼                     │
                    STOPPING  <──stop()──┬────────┤
                       │                 │        │
                       ▼                 │        │
                    STOPPED <────────────┘        │
                       │    (正常停止)            │
                       │                          │
                       ▼                          │
                    ERROR <───────────────────────┘
                    (异常终止)

    终态: STOPPED, ERROR
    可重启: 从终态可以重新调用 start()

【ReplayPipeline 状态流转】

    IDLE ──start()──> STARTING ──> RUNNING ──文件读完──> COMPLETED
                       │                              (正常完成)
                       │
                       ├──────────stop()──────────> STOPPED
                       │                              (手动停止)
                       │
                       └──────────异常────────────> ERROR
                                                      (异常终止)

    终态: COMPLETED, STOPPED, ERROR
    可重启: 从终态可以重新调用 start()
    """)


def show_api_differences():
    """展示 API 差异"""
    
    print("\n" + "=" * 70)
    print("API 差异")
    print("=" * 70)
    
    print("""
【共有 API】

    pipeline.start()           # 启动 Pipeline（非阻塞）
    pipeline.stop()            # 停止 Pipeline
    pipeline.run()             # 阻塞式运行
    pipeline.wait(timeout)     # 等待进入终态
    pipeline.get_state()       # 获取当前状态
    pipeline.get_progress()    # 获取进度信息
    pipeline.is_running()      # 是否运行中
    pipeline.is_stopped()      # 是否已停止
    pipeline.is_error()        # 是否出错
    pipeline.is_terminal()     # 是否处于终态

【ReplayPipeline 特有 API】

    pipeline.is_completed()    # 是否正常完成

【get_progress() 返回差异】

LivePipeline.get_progress() 返回:
    - state: 当前状态
    - processed_frames: 已处理帧数
    - start_time: 开始时间
    - current_timestamp: 当前时间戳
    - error: 错误信息

ReplayPipeline.get_progress() 额外返回:
    - total_frames: 总帧数（如可获取）
    - duration: 总时长（秒）
    """)


def show_best_practices():
    """展示最佳实践"""
    
    print("\n" + "=" * 70)
    print("最佳实践")
    print("=" * 70)
    
    print("""
【选择 Pipeline 的决策树】

    输入源是实时流？
    ├── 是 → 使用 LivePipeline
    │         └── 需要持续运行监控
    │
    └── 否 → 输入源是文件？
              ├── 是 → 使用 ReplayPipeline
              │         ├── 需要进度跟踪？→ 启用详细日志
              │         └── 需要快速预览？→ keyframe_only 模式
              │
              └── 否 → 检查输入源类型

【VLM 后端选择建议】

    1. 入门学习 / 调试验证
       → 使用 BMP 后端（无需 GPU，保存图像为 BMP 文件）
       
    2. 本地 GPU 推理
       → 使用 Qwen3.5 后端（显存占用低，约 1.4GB）
       → 安装: pip install 'pymediaparser[vlm-qwen]'
       
    3. 远程 API 服务
       → 使用 OpenAI API 后端（vLLM / Ollama / OpenAI）
       → 无需本地 GPU

【性能优化建议】

1. 实时流场景 (LivePipeline)
   - 设置合理的 max_queue_size（默认 3）
   - 启用智能采样减少帧处理
   - 根据网络状况调整 reconnect_interval
   - 无 GPU 时使用 BMP 后端调试

2. 文件回放场景 (ReplayPipeline)
   - 启用批量处理提升吞吐
   - 仅关键帧模式加速预览
   - 智能采样减少冗余分析
   - 大文件分片并行处理

【错误处理建议】

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        if pipeline.is_running():
            pipeline.stop()
            pipeline.wait(timeout=10.0)
        
        # 检查最终状态
        if pipeline.is_error():
            print("Pipeline 异常终止")
        elif pipeline.is_completed():
            print("Pipeline 正常完成")
        elif pipeline.is_stopped():
            print("Pipeline 被手动停止")
    """)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LivePipeline vs ReplayPipeline 对比示例")
    print("=" * 70)
    print("\n【VLM 后端选择指南】")
    print("-" * 70)
    print("| 后端       | 说明                          | 安装要求          |")
    print("|------------|-------------------------------|-------------------|")
    print("| bmp        | 虚拟调试，保存图像为 BMP      | 核心包（推荐入门）|")
    print("| qwen35     | Qwen3.5 本地 GPU 推理         | [vlm-qwen]        |")
    print("| openai_api | vLLM/Ollama/OpenAI API 服务   | 核心包            |")
    print("-" * 70)
    
    show_comparison()
    show_use_cases()
    show_code_comparison()
    show_state_machine()
    show_api_differences()
    show_best_practices()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
核心要点:

1. LivePipeline 用于实时流，ReplayPipeline 用于文件
2. 两者 API 高度一致，切换成本低
3. 共享相同的配置类和 VLM 客户端
4. 都支持智能采样和批量处理优化
5. 都支持多种 VLM 后端（BMP/Qwen3.5/OpenAI API）

选择建议:

- 入门学习 → BMP 后端（无需 GPU）
- 实时监控/直播 → LivePipeline
- 离线分析/批量处理 → ReplayPipeline
- 本地 GPU 推理 → Qwen3.5 后端
- 远程 API 服务 → OpenAI API 后端
- 不确定时 → 先用 BMP 后端 + ReplayPipeline 测试文件
    """)
