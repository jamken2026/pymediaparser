# pymediaparser
 
媒体流解析与视觉语言模型工具包，提供实时视频流接入、智能帧采样、VLM 推理等功能。

## 功能特性

- **实时流接入**：支持 RTMP/HTTP-FLV/HTTP-TS 等协议的视频流拉取
- **智能帧采样**：基于运动检测和内容变化分析的自适应采样，减少无效帧处理
- **VLM 推理**：集成 Qwen2-VL 视觉语言模型，支持图像理解和问答
- **批量处理**：支持帧缓冲和批量推理，提升 GPU 利用率
- **灵活扩展**：模块化设计，支持自定义采样器和结果处理器

## 安装

### 基础安装

```bash
pip install pymediaparser
```

### VLM 支持

如需使用视觉语言模型功能：

```bash
pip install pymediaparser[vlm]
```

### 开发环境

```bash
pip install pymediaparser[dev]
```

## 快速开始

### 方式 1：命令行脚本

使用 `scripts/run_parser.py` 脚本快速启动媒体流解析：

```bash
# RTMP 流，每秒 1 帧，使用 GPU 推理
python scripts/run_parser.py --url rtmp://host/live/stream

# HTTP-FLV 流，自定义 prompt
python scripts/run_parser.py \
    --url http://host/live/stream.flv \
    --fps 0.5 \
    --prompt "识别画面中的人物并描述他们的行为。"

# 启用智能采样模式
python scripts/run_parser.py \
    --url rtmp://host/live/stream \
    --smart-sampling \
    --ssim-threshold 0.8

# 启用批量处理模式
python scripts/run_parser.py \
    --url rtmp://host/live/stream \
    --batch-processing

# 组合使用智能采样 + 批量处理
python scripts/run_parser.py \
    --url rtmp://host/live/stream \
    --smart-sampling \
    --batch-processing

# CPU 模式（无 GPU 时）
python scripts/run_parser.py \
    --url rtmp://host/live/stream \
    --device cpu --dtype float32 --fps 0.2
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--url` | 必填 | 视频流地址 (RTMP/HTTP-FLV/HTTP-TS) |
| `--fps` | 1.0 | 目标抽帧频率 |
| `--device` | cuda:0 | 推理设备 |
| `--prompt` | 内置默认 | VLM 提示词 |
| `--model-path` | 内置模型 | Qwen2-VL 模型路径 |
| `--smart-sampling` | False | 启用智能采样 |
| `--batch-processing` | False | 启用批量处理 |
| `--callback-url` | None | HTTP 回调地址 |

### 方式 2：Python API

#### 阻塞模式（简单场景）

```python
from pymediaparser import LivePipeline, StreamConfig, VLMConfig, Qwen2VLClient

# 配置视频流
stream_cfg = StreamConfig(url="rtmp://host/live/stream", target_fps=1.0)

# 配置 VLM 客户端
vlm_client = Qwen2VLClient(VLMConfig(device="cuda:0"))

# 创建并运行 Pipeline（阻塞直到 Ctrl+C）
pipeline = LivePipeline(stream_cfg, vlm_client, prompt="请描述画面中的内容。")
pipeline.run()
```

#### 异步模式（外部系统集成）

适用于外部系统调用，支持启动后立即返回、进度查询和主动停止：

```python
import time
from pymediaparser import (
    ReplayPipeline, LivePipeline, StreamConfig,
    PipelineState, create_vlm_client
)
from pymediaparser.result_handler import ResultHandler

# 自定义 Handler（可选）
class MyHandler(ResultHandler):
    def handle(self, result):
        # 处理结果：写入数据库、发送消息等
        save_to_db(result)
    
    def on_complete(self):
        print("Pipeline 正常完成")
    
    def on_error(self, error):
        print(f"Pipeline 异常: {error}")

# 创建 Pipeline
stream_cfg = StreamConfig(url="/path/to/video.mp4", target_fps=1.0)
vlm_client = create_vlm_client("openai_api", {"base_url": "http://localhost:8000/v1"})
pipeline = ReplayPipeline(stream_cfg, vlm_client, handlers=[MyHandler()])

# 启动（立即返回）
pipeline.start()

# 轮询进度
while pipeline.is_running():
    progress = pipeline.get_progress()
    if progress.duration:
        # ReplayPipeline: 显示百分比
        print(f"进度: {progress.progress_percent:.1f}% | "
              f"{progress.current_timestamp:.1f}s / {progress.duration:.1f}s")
    else:
        # LivePipeline: 显示帧数和运行时间
        print(f"已处理: {progress.processed_frames} 帧 | "
              f"运行时间: {progress.elapsed_time:.0f}s")
    time.sleep(5)

# 检查最终状态（资源已清理，无需调用 stop()）
match pipeline.get_state():
    case PipelineState.COMPLETED:
        print("处理完成")
    case PipelineState.STOPPED:
        print("被主动停止")
    case PipelineState.ERROR:
        print(f"异常: {pipeline.get_progress().error}")
```

#### 主动停止 Pipeline

```python
# 启动 Pipeline
pipeline.start()

# 运行一段时间后主动停止
time.sleep(60)
pipeline.stop()  # 优雅停止，清理资源
```

#### Pipeline 状态说明

| 状态 | 含义 | 资源状态 |
|------|------|----------|
| `IDLE` | 初始状态，未启动 | 未分配 |
| `STARTING` | 启动中（加载模型） | 部分分配 |
| `RUNNING` | 正常运行中 | 已分配 |
| `STOPPING` | 停止中（清理资源） | 清理中 |
| `COMPLETED` | 正常完成（仅 ReplayPipeline） | 已清理 |
| `STOPPED` | 被主动停止 | 已清理 |
| `ERROR` | 异常终止 | 已清理 |

**核心契约**：进入任何终态（COMPLETED/STOPPED/ERROR）时，资源已清理完毕，调用者无需额外操作。

#### 智能模式（自适应采样）

```python
from pymediaparser import LivePipeline, StreamConfig, VLMConfig, Qwen2VLClient

stream_cfg = StreamConfig(url="rtmp://host/live/stream", target_fps=30.0)
vlm_client = Qwen2VLClient(VLMConfig(device="cuda:0"))

# 启用智能采样
pipeline = LivePipeline(
    stream_cfg,
    vlm_client,
    prompt="请描述画面中的内容。",
    enable_smart_sampling=True,
    smart_config={
        'motion_threshold': 0.3,      # 运动检测阈值
        'backup_interval': 30.0,      # 保底采样间隔（秒）
        'min_frame_interval': 1.0,    # 最小帧间隔（秒）
    },
)
pipeline.run()
```
 
## 主要组件

### Pipeline

| 组件 | 说明 |
|------|------|
| `LivePipeline` | 实时流 VLM 分析 Pipeline |
| `ReplayPipeline` | 文件回放 VLM 分析 Pipeline |
| `PipelineState` | Pipeline 运行状态枚举 |
| `PipelineProgress` | Pipeline 进度信息 |

### 流接入

| 组件 | 说明 |
|------|------|
| `StreamReader` | 实时流解码器，支持 RTMP/HTTP-FLV/HTTP-TS |

### 帧采样

| 组件 | 说明 |
|------|------|
| `FrameSampler` | 固定频率采样器 |
| `SimpleSmartSampler` | 简单智能采样器，基于运动检测和 SSIM 变化分析 |
| `MLSmartSampler` | 多层漏斗型智能采样器，三层过滤架构 |

### VLM 客户端

| 组件 | 说明 |
|------|------|
| `VLMClient` | VLM 客户端抽象基类 |
| `Qwen2VLClient` | Qwen2-VL 模型客户端实现 |

### 结果处理

| 组件 | 说明 |
|------|------|
| `ResultHandler` | 结果处理器抽象基类 |
| `ConsoleResultHandler` | 控制台输出处理器 |
| `HttpCallbackHandler` | HTTP 回调处理器 |

### 配置类

| 组件 | 说明 |
|------|------|
| `StreamConfig` | 流配置（URL、帧率、队列大小等） |
| `VLMConfig` | VLM 配置（模型路径、设备等） |

## 智能采样器对比

| 特性 | FrameSampler | SimpleSmartSampler | MLSmartSampler |
|------|--------------|-------------------|----------------|
| 采样策略 | 固定频率 | 运动检测 + 变化分析 | 三层漏斗过滤 |
| 计算开销 | 无 | 低 | 中 |
| 过滤精度 | 无 | 中 | 高 |
| 适用场景 | 均匀采样 | 一般智能采样 | 高精度场景 |

## 依赖

- Python >= 3.10
- PyTorch
- Transformers (VLM 功能)
- OpenCV
- NumPy
- Pillow
- PyAV

## 许可证

MIT License
