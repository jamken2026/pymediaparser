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

#### 传统模式（固定频率抽帧）

```python
from pymediaparser import LivePipeline, StreamConfig, VLMConfig, Qwen2VLClient

# 配置视频流
stream_cfg = StreamConfig(url="rtmp://host/live/stream", target_fps=1.0)

# 配置 VLM 客户端
vlm_client = Qwen2VLClient(VLMConfig(device="cuda:0"))

# 创建并运行 Pipeline
pipeline = LivePipeline(stream_cfg, vlm_client, prompt="请描述画面中的内容。")
pipeline.run()
```

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
        'ssim_threshold': 0.80,       # 变化检测阈值
        'backup_interval': 30.0,      # 保底采样间隔（秒）
        'min_frame_interval': 1.0,    # 最小帧间隔（秒）
    },
)
pipeline.run()
```
 
## 主要组件

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
