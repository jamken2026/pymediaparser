# Replay Pipeline 实现方案

## 概述

新增回放（点播）模式，通过读取视频文件/图片文件进行 VLM 分析。核心变更：新建 `FileReader` + `ReplayPipeline` 两个模块，修改 `run_parser.py` 增加 `--mode` 路由。

---

## 新增文件

### 1. `src/pymediaparser/file_reader.py` - 文件读取器

独立的 `FileReader` 类，与 `StreamReader` 平级，不共享继承关系。

**职责**：打开本地/网络视频文件或图片，以 `(av.VideoFrame, float)` 元组迭代输出帧。

**核心接口**（与 StreamReader 对齐风格）：

```python
class FileReader:
    def __init__(self, config: StreamConfig) -> None
    def open(self) -> None              # 打开文件，收集元信息
    def frames(self) -> Iterator[tuple[av.VideoFrame, float]]  # 逐帧迭代
    def close(self) -> None             # 释放资源
    def __enter__ / __exit__            # 上下文管理器

    # 进度信息属性
    @property
    def total_frames(self) -> int       # 视频总帧数（0=未知），图片=1
    @property
    def duration_seconds(self) -> float # 视频总时长秒（0.0=未知），图片=0.0
    @property
    def is_image(self) -> bool          # 是否为图片文件
```

**文件类型检测** (`_detect_source_type()`):
- 从 URL/路径提取扩展名
- 图片扩展名集合：`{bmp, jpg, jpeg, png, gif, webp, tiff, tif}`
- 其余（mp4, mov, m3u8, ts, ps, flv, avi, mkv 等）→ 视频类型
- 无扩展名默认视频（让 PyAV 自动探测）

**视频处理路径**:
- 使用 `av.open(url_or_path)` 打开，不设低延迟参数（去掉 `nobuffer`/`low_delay`）
- 对网络 URL 保留合理的 `timeout`
- `frames()` 迭代 `container.decode(video=0)`，与 StreamReader 输出格式一致
- 文件读完后 `frames()` 自然结束（不重连）
- 元信息：通过 `stream.frames`（总帧数）和 `container.duration`（总时长微秒）获取

**图片处理路径**:
- 本地图片：`PIL.Image.open(path)`
- 网络图片：`urllib.request.urlopen(url)` 读入 `io.BytesIO` 后 `PIL.Image.open()`
- 转为 `av.VideoFrame`：PIL → numpy(rgb24) → `av.VideoFrame.from_ndarray(arr, format="rgb24")`
- `frames()` yield 单帧 `(video_frame, 0.0)` 后结束

**关键点**：输出格式 `(av.VideoFrame, float)` 与 `StreamReader.frames()` 完全一致，`FrameSampler.sample()` 无需任何修改即可对接。

---

### 2. `src/pymediaparser/replay_pipeline.py` - 回放 Pipeline

镜像 `LivePipeline` 的架构（2 线程 / 3 线程），核心差异为队列策略和完成检测。

**构造函数参数与 LivePipeline 完全一致**:
```python
class ReplayPipeline:
    def __init__(
        self,
        stream_config: StreamConfig,
        vlm_client: VLMClient,
        handlers: Sequence[ResultHandler] | None = None,
        prompt: str | None = None,
        enable_smart_sampling: bool = False,
        enable_batch_processing: bool = False,
        smart_config: Optional[Dict[str, Any]] = None,
    ) -> None
```

#### 2.1 队列策略差异（核心变更）

| 操作 | LivePipeline | ReplayPipeline |
|------|-------------|----------------|
| 生产者入队 | `get_nowait()` 丢旧 + `put_nowait()` | `put(block=True, timeout=0.5)` 短超时阻塞 |
| 处理器入队 `_enqueue_frame()` | 同上丢旧 | `put(block=True, timeout=0.5)` 短超时阻塞 |
| 消费者出队 | `get(timeout=1.0)` | 相同 |

生产者内入队逻辑（伪代码）：
```python
while not _stop_event.is_set():
    try:
        target_queue.put(item, timeout=0.5)
        break  # 入队成功
    except queue.Full:
        continue  # 0.5s 超时后立即重新检查 stop_event，保证及时响应退出
```

> **设计说明**：put 超时设为 0.5s 而非更长值，确保文件读完/异常/Ctrl+C 时线程能在 1 秒内感知并退出，避免不必要的等待。

#### 2.2 完成检测机制

新增 `_done_event: threading.Event`：
- 消费者线程收到 sentinel (None) 退出前设置 `_done_event.set()`
- `run()` 等待 `_done_event` 或 `KeyboardInterrupt`：

```python
def run(self) -> None:
    self.start()
    try:
        while not self._done_event.is_set() and not self._stop_event.is_set():
            self._done_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在停止 ...")
    finally:
        self.stop()
```

#### 2.3 Sentinel 传播

**传统模式（2 线程）**：
```
FileReader.frames() 迭代完毕
  → 生产者 finally 向 _queue 发 None
  → 消费者收到 None → flush 缓冲区 → _done_event.set() → 退出
```

**智能模式（3 线程）**：
```
FileReader.frames() 迭代完毕
  → 生产者 finally 向 _processor_queue 发 None
  → 处理器收到 None → 向 _queue 发 None → 退出
  → 消费者收到 None → flush 缓冲区 → _done_event.set() → 退出
```

生产者 finally 块发送 sentinel 也使用短超时 put：
```python
finally:
    # 发送 sentinel 通知下游（短超时，避免卡死）
    try:
        target_queue.put(None, timeout=1.0)
    except queue.Full:
        pass
    file_reader.close()
```

#### 2.4 进度跟踪

在 ReplayPipeline 中维护进度状态：
- `_file_reader: FileReader` 引用（获取 total_frames / duration_seconds）
- `_processed_count: int` 原子计数器（消费者每处理一帧递增）

消费者 `_process_frame_item()` 处理完帧后调用 `_log_progress(item)`：
```
# 已知总帧数时：
[回放进度] 帧 5/120 | 视频 15.0s/360.0s | 4.2%

# 未知总帧数时：
[回放进度] 已处理 5 帧 | 视频 15.0s
```

进度通过 `logger.info` 输出，不单独引入新的 handler。

#### 2.5 stop() 方法调整

由于阻塞 put，stop() 时线程可能阻塞在 `queue.put()` 上。处理方式：
1. `_stop_event.set()` → 生产者/处理器的 put 在 0.5s 超时后检测到 stop 并退出
2. 同时向队列放 sentinel（`put_nowait`），让消费者也能退出
3. `join(timeout=3)` 等待线程（put 超时仅 0.5s，线程能快速响应退出）

#### 2.6 可复用逻辑（从 LivePipeline 复制）

以下方法直接复制，逻辑不变：
- `_consumer_loop()` - 消费者主循环（相同，但退出前设 `_done_event`）
- `_process_frame_item()` - 单帧/批处理推理
- `_process_batch()` - 批量推理
- `_build_batch_prompt()` - 批处理提示词
- `_flush_batch_buffer()` - 清空缓冲区
- `_check_batch_timeout()` - 批处理超时检查
- `_processor_loop()` - 处理器主循环（`_enqueue_frame` 改为阻塞版本）
- `_numpy_to_pil()` - 格式转换

**不同的方法**：
- `__init__()` - 初始化 `_done_event`，`_file_reader` 代替 `_stream_reader`
- `_producer_loop()` - 使用 `FileReader` + 阻塞 put + 进度注入
- `_enqueue_frame()` - 阻塞 put 版本
- `start()` - 日志输出调整
- `stop()` - 适配阻塞 put 的停止
- `run()` - 等待 `_done_event` 而非无限循环

---

## 修改文件

### 3. `scripts/run_parser.py` - 入口脚本

**新增参数**：
```python
parser.add_argument(
    "--mode", default="live", choices=["live", "replay"],
    help="运行模式: live=实时流 / replay=文件回放（默认: live）",
)
```

放在最顶层（非分组），作为模式选择。

**`--url` help 文本扩展**：
```
help="live: 流地址 (rtmp:// / http://*.flv 等); replay: 文件路径或URL (/path/to/video.mp4)"
```

**Pipeline 构建分支**（在 `main()` 末尾）：
```python
if args.mode == "replay":
    from pymediaparser.replay_pipeline import ReplayPipeline
    pipeline = ReplayPipeline(
        stream_config=stream_cfg,
        vlm_client=vlm_client,
        handlers=handlers,
        prompt=args.prompt,
        enable_smart_sampling=args.smart_sampling,
        enable_batch_processing=args.batch_processing,
        smart_config=smart_config,
    )
else:
    pipeline = LivePipeline(...)  # 现有逻辑不变
```

**日志信息适配**：
- mode=replay 时打印"文件回放 VLM 分析"而非"实时流 VLM 分析"
- 不显示"断线重连"
- 显示"运行模式: 文件回放"

### 4. `src/pymediaparser/__init__.py` - 包导出

在 `__all__` 中追加：
```python
'FileReader',
'ReplayPipeline',
```

在 `__getattr__` 的 `_import_map` 中追加：
```python
'FileReader':       ('.file_reader', 'FileReader'),
'ReplayPipeline':   ('.replay_pipeline', 'ReplayPipeline'),
```

---

## 文件清单

| 操作 | 文件路径 |
|------|---------|
| 新增 | `src/pymediaparser/file_reader.py` |
| 新增 | `src/pymediaparser/replay_pipeline.py` |
| 修改 | `scripts/run_parser.py` |
| 修改 | `src/pymediaparser/__init__.py` |
| 新增 | `tests/test_file_reader.py` |
| 新增 | `tests/test_replay_pipeline.py` |

不修改：`live_pipeline.py`、`stream_reader.py`、`frame_sampler.py`、`vlm_base.py` 等现有文件。

---

## 验证方案

测试资源（已有）：
- 视频：`resource/866f893bb7a353c75f1aa5c7cb61e4e3.mp4`
- 图片：`resource/test_img1.png`、`resource/test_img2.png`

验证脚本放在 `tests/` 目录下（遵循项目规范）：

### 新增测试文件：`tests/test_file_reader.py`

使用 pytest 编写，测试 FileReader 的核心功能：
- 视频文件（mp4）：`frames()` 输出格式为 `(av.VideoFrame, float)`，帧数 > 0
- 图片文件（png）：`frames()` 输出单帧，`is_image == True`
- 属性验证：`total_frames` / `duration_seconds` / `is_image`
- 不存在的文件：合理抛异常

### 新增测试文件：`tests/test_replay_pipeline.py`

使用 pytest + mock VLM 客户端编写，测试 ReplayPipeline 的核心功能：
- 视频回放：使用 mp4 文件，mock VLM 客户端，验证 pipeline 自动完成退出（不挂起）
- 图片回放：单张图片，pipeline 处理 1 帧后自动退出
- 阻塞队列验证：所有采样帧均被处理（不丢帧）
- 进度日志输出验证
- 智能采样模式（3 线程）正常工作

### CLI 手动验证命令

```bash
# 基础回放（传统模式）
python scripts/run_parser.py --mode replay \
    --url resource/866f893bb7a353c75f1aa5c7cb61e4e3.mp4 \
    --vlm-backend openai_api --api-base-url http://localhost:8000/v1

# 图片分析
python scripts/run_parser.py --mode replay \
    --url resource/test_img1.png \
    --vlm-backend openai_api --api-base-url http://localhost:8000/v1

# 默认模式不受影响
python scripts/run_parser.py --url rtmp://... # 仍为 live 模式
```
