# VLM 图像分析器灵活模型切换方案

## 目标
将图像分析器从 Qwen2-VL 单一模型扩展为支持多种 VLM 后端：
- **本地 transformers 模型**（Qwen2-VL、Qwen3-VL 等）
- **OpenAI 兼容 API 服务**（vLLM、Ollama、OpenAI、通义千问等）

以新增 Qwen3-VL-2B 支持作为首个验证用例（模型已下载到本地）。

## 架构审查：发现的问题与修正

### 问题 1: Qwen2/Qwen3 大量代码重复
`Qwen2VLClient`（337 行）和 `Qwen3VLClient` 共享 ~80% 代码：config 处理、dtype 映射、Flash Attention 回退、GPU 管理、`torch.inference_mode()` + `generate()` + decode、`unload()`、`VLMResult` 构造。仅模型加载类和输入预处理不同。

**修正**：抽取 `_LocalTransformersBase` 公共基类，子类只需覆盖 `_load_model_and_processor()` 和 `_prepare_inputs()`。未来添加 InternVL、LLaVA 等也只需实现这两个方法。

### 问题 2: 配置类层级过度设计
原方案 `BaseVLMConfig → LocalVLMConfig` + `BaseVLMConfig → APIVLMConfig` 三层继承，但 `BaseVLMConfig` 仅有 2 个字段（`max_new_tokens`, `default_prompt`），与 `VLMConfig` 字段重复。

**修正**：去掉 `BaseVLMConfig`。`LocalVLMConfig` 直接继承现有 `VLMConfig`（空子类，零重复）。`APIVLMConfig` 独立定义。

### 问题 3: `live_pipeline.py` 有两处 Qwen2VLClient 硬编码
- 第 31 行: `from .vlm_qwen2 import Qwen2VLClient`（import）
- 第 609 行: `vlm_client = Qwen2VLClient(vlm_cfg)`（`main()` 函数中实例化）

**修正**：`main()` 函数也改用工厂函数。

### 问题 4: `_DTYPE_MAP` 等工具代码会重复
dtype 映射、GPU 信息打印等在 Qwen2/Qwen3 中完全相同。

**修正**：提取到 `vlm/_local_base.py` 的公共基类中。

### 问题 5: 工厂延迟导入机制未明确
原方案写 `lazy_import(...)` 但未说明实现。

**修正**：使用 `(module_path, class_name)` 字符串元组 + `importlib.import_module()` 按需加载。

### 问题 6: `analyze_with_context` / `_build_context_prompt` 未纳入设计
这些是 `Qwen2VLClient` 的额外方法，不在 `VLMClient` ABC 中。Qwen3 和 API 后端也需要它们。

**修正**：移入 `_LocalTransformersBase`（本地后端共享），API 后端在 `OpenAIAPIClient` 中单独实现。

---

## 最终包结构

```
src/pymediaparser/
├── vlm_base.py                  # 不动：VLMClient ABC, VLMConfig, VLMResult, StreamConfig, FrameResult
├── vlm_qwen2.py                 # 改为 re-export: from .vlm.qwen2 import Qwen2VLClient
├── vlm/                         # 新建子包
│   ├── __init__.py              # re-export 公开符号
│   ├── configs.py               # LocalVLMConfig(继承VLMConfig), APIVLMConfig(独立)
│   ├── factory.py               # create_vlm_client(), register_vlm_backend()
│   ├── _local_base.py           # _LocalTransformersBase（本地模型公共逻辑）
│   ├── qwen2.py                 # Qwen2VLClient（从 vlm_qwen2.py 迁入，继承 _LocalTransformersBase）
│   ├── qwen3.py                 # Qwen3VLClient（新建，继承 _LocalTransformersBase）
│   └── openai_api.py            # OpenAIAPIClient（新建，直接继承 VLMClient）
├── smart_sampler/               # 不变
├── live_pipeline.py             # 修改：删除硬编码 import 和 main() 中的直接实例化
└── ...
```

---

## 实现方案

### 1. 新建 `vlm/configs.py` — 配置类

```python
from ..vlm_base import VLMConfig

@dataclass
class LocalVLMConfig(VLMConfig):
    """本地 transformers 模型配置。继承 VLMConfig，保持完全兼容。"""
    pass  # 字段全部继承自 VLMConfig，零重复

@dataclass
class APIVLMConfig:
    """OpenAI 兼容 API 配置。"""
    base_url: str                           # 如 http://localhost:8000
    model_name: str                         # 如 gpt-4o, Qwen2-VL-7B-Instruct
    api_key: str | None = None
    max_new_tokens: int = 256
    default_prompt: str = "请描述当前画面中的人物活动。"
    timeout: float = 30.0
    max_retries: int = 2
    image_max_size: int = 1024              # base64 前 resize 上限（像素）
    image_quality: int = 85                 # JPEG 编码质量
```

**关系图**：
```
VLMConfig (vlm_base.py, 不动)
└── LocalVLMConfig (vlm/configs.py, 空子类)

APIVLMConfig (vlm/configs.py, 独立 dataclass)
```

### 2. 新建 `vlm/_local_base.py` — 本地模型公共基类

抽取 Qwen2/Qwen3 共享的 ~200 行逻辑：

```python
class _LocalTransformersBase(VLMClient):
    """本地 transformers 模型的公共推理逻辑。"""

    # 共享的 dtype 映射
    _DTYPE_MAP = {"float16": torch.float16, "fp16": torch.float16, ...}

    def __init__(self, config: VLMConfig | None = None):
        self.config = config or VLMConfig()
        self._model = None
        self._processor = None

    # --- 子类必须覆盖的两个方法 ---

    @abstractmethod
    def _load_model_and_processor(self, cfg, load_kwargs) -> tuple[Any, Any]:
        """加载模型和处理器，返回 (model, processor)。"""

    @abstractmethod
    def _prepare_inputs(self, messages) -> dict:
        """将多模态消息转为模型输入 tensor dict。"""

    # --- 公共实现（子类无需覆盖） ---

    def load(self):
        cfg = self.config
        dtype = self._DTYPE_MAP.get(cfg.dtype, torch.float16)
        load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True, ...}
        # Flash Attention 回退逻辑（通用）
        # 调用子类 _load_model_and_processor
        self._model, self._processor = self._load_model_and_processor(cfg, load_kwargs)
        # CPU 模式移动、model.eval()、GPU 信息打印

    def analyze(self, image, prompt=None):
        # 构造 messages
        # 调用子类 _prepare_inputs(messages)
        # torch.inference_mode() + generate()
        # decode + 构造 VLMResult

    def analyze_batch(self, images, prompt):
        # 构造多图 messages
        # 调用子类 _prepare_inputs(messages)
        # generate() + decode

    def unload(self):
        # del model/processor + torch.cuda.empty_cache()

    def analyze_with_context(self, images, context_prompt=None):
        # 共享的上下文分析逻辑

    def _build_context_prompt(self, image_count):
        # 共享的 prompt 构造
```

### 3. 迁移 `vlm_qwen2.py` → `vlm/qwen2.py`

重构为继承 `_LocalTransformersBase`，**只保留差异部分**：

```python
class Qwen2VLClient(_LocalTransformersBase):
    """Qwen2-VL 本地推理客户端。"""

    def _load_model_and_processor(self, cfg, load_kwargs):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_path, **load_kwargs)
        processor = AutoProcessor.from_pretrained(cfg.model_path, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
        return model, processor

    def _prepare_inputs(self, messages):
        from qwen_vl_utils import process_vision_info
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        return inputs
```

原 `vlm_qwen2.py` 改为薄 re-export：
```python
from .vlm.qwen2 import Qwen2VLClient  # noqa: F401 — 向后兼容
```

### 4. 新建 `vlm/qwen3.py` — Qwen3-VL 客户端

继承同一基类，仅 2 个方法不同：

```python
class Qwen3VLClient(_LocalTransformersBase):
    """Qwen3-VL 本地推理客户端。"""

    def _load_model_and_processor(self, cfg, load_kwargs):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(cfg.model_path, **load_kwargs)
        processor = AutoProcessor.from_pretrained(cfg.model_path, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
        return model, processor

    def _prepare_inputs(self, messages):
        # Qwen3-VL 一步到位，不需要 process_vision_info
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        return inputs
```

### 5. 新建 `vlm/openai_api.py` — OpenAI 兼容 API 客户端

直接继承 `VLMClient`（不经过 `_LocalTransformersBase`）：

```python
class OpenAIAPIClient(VLMClient):
    def __init__(self, config: APIVLMConfig): ...

    def load(self):
        self._session = requests.Session()
        if self.config.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.config.api_key}"

    def analyze(self, image, prompt=None):
        base64_image = self._encode_image(image)
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt or self.config.default_prompt}
            ]}],
            "max_tokens": self.config.max_new_tokens
        }
        resp = self._request_with_retry(payload)
        return self._parse_response(resp)

    def analyze_batch(self, images, prompt): ...  # 多图放入同一 content 数组
    def unload(self): self._session.close()
    def _encode_image(self, image, max_size=None, quality=None): ...
    def _request_with_retry(self, payload): ...  # 指数退避重试
    def _parse_response(self, resp): ...          # 提取 text → VLMResult
```

### 6. 新建 `vlm/factory.py` — 工厂函数和注册表

```python
import importlib

# 字符串元组注册表 → 不触发任何 import
_REGISTRY: dict[str, tuple[str, str]] = {
    'qwen2-vl':   ('pymediaparser.vlm.qwen2',      'Qwen2VLClient'),
    'qwen3-vl':   ('pymediaparser.vlm.qwen3',      'Qwen3VLClient'),
    'openai-api': ('pymediaparser.vlm.openai_api',  'OpenAIAPIClient'),
    'vllm':       ('pymediaparser.vlm.openai_api',  'OpenAIAPIClient'),
    'ollama':     ('pymediaparser.vlm.openai_api',  'OpenAIAPIClient'),
}

def create_vlm_client(backend: str, config) -> VLMClient:
    """根据后端名称创建 VLM 客户端。"""
    if backend not in _REGISTRY:
        raise ValueError(f"未知后端 '{backend}'，可用: {list(_REGISTRY.keys())}")
    module_path, class_name = _REGISTRY[backend]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)

def register_vlm_backend(name: str, module_path: str, class_name: str):
    """注册自定义 VLM 后端（延迟导入）。"""
    _REGISTRY[name] = (module_path, class_name)

def list_backends() -> list[str]:
    """列出所有可用后端名称。"""
    return list(_REGISTRY.keys())
```

**关键**：注册表只存字符串，`import transformers` 仅在实际调用 `create_vlm_client('qwen2-vl', ...)` 时才发生。API-only 用户完全不会触发 transformers 导入。

### 7. 新建 `vlm/__init__.py` — 子包入口

```python
from .configs import LocalVLMConfig, APIVLMConfig
from .factory import create_vlm_client, register_vlm_backend, list_backends
# 客户端类不在此导入 — 通过 factory 延迟加载
```

### 8. 修改 `__init__.py` — 更新延迟导入映射

```python
# 旧映射更新（指向新位置）
'Qwen2VLClient':         ('.vlm.qwen2', 'Qwen2VLClient'),
# 新增映射
'LocalVLMConfig':         ('.vlm.configs', 'LocalVLMConfig'),
'APIVLMConfig':           ('.vlm.configs', 'APIVLMConfig'),
'Qwen3VLClient':          ('.vlm.qwen3', 'Qwen3VLClient'),
'OpenAIAPIClient':        ('.vlm.openai_api', 'OpenAIAPIClient'),
'create_vlm_client':      ('.vlm.factory', 'create_vlm_client'),
'register_vlm_backend':   ('.vlm.factory', 'register_vlm_backend'),
```

`__all__` 列表同步更新。

### 9. 修改 `vlm_base.py` — 最小改动

- `DEFAULT_MODEL_PATH` 环境变量名改为更通用的 `VLM_MODEL_PATH`，保留 `QWEN_VL_MODEL_PATH` 作为回退：
  ```python
  DEFAULT_MODEL_PATH = os.environ.get(
      "VLM_MODEL_PATH",
      os.environ.get("QWEN_VL_MODEL_PATH", os.path.join(...)),
  )
  ```
- `VLMConfig` 保持不变

### 10. 修改 `live_pipeline.py`

- 删除第 31 行: `from .vlm_qwen2 import Qwen2VLClient`
- 修改 `main()` 函数（第 609 行）：用 `create_vlm_client('qwen2-vl', vlm_cfg)` 替代 `Qwen2VLClient(vlm_cfg)`

### 11. 修改 `scripts/run_parser.py` — CLI 集成

新增参数：
```
--vlm-backend     后端类型 (qwen2-vl|qwen3-vl|openai-api|vllm|ollama)，默认 qwen2-vl
--api-base-url    API 基础 URL（API 后端必填）
--api-key         API 密钥（可选）
--api-model       API 模型名称（API 后端必填）
```

逻辑分支：
```python
if args.vlm_backend in ('openai-api', 'vllm', 'ollama'):
    config = APIVLMConfig(base_url=args.api_base_url, model_name=args.api_model, api_key=args.api_key, ...)
else:
    config = VLMConfig(model_path=args.model_path, device=args.device, ...)
vlm_client = create_vlm_client(args.vlm_backend, config)
```

### 12. 修改 `pyproject.toml`

```toml
[project.optional-dependencies]
vlm = ["transformers>=4.40", "qwen-vl-utils>=0.0.8"]
vlm-qwen3 = ["transformers>=4.57.0", "qwen-vl-utils>=0.0.8"]
```

---

## 类继承关系总览

```
VLMClient (ABC, vlm_base.py)
├── _LocalTransformersBase (vlm/_local_base.py)
│   ├── Qwen2VLClient     (vlm/qwen2.py)
│   └── Qwen3VLClient     (vlm/qwen3.py)
└── OpenAIAPIClient        (vlm/openai_api.py)

VLMConfig (dataclass, vlm_base.py)
└── LocalVLMConfig         (vlm/configs.py, 空子类)

APIVLMConfig               (vlm/configs.py, 独立 dataclass)
```

---

## 文件变更清单

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| **新建** | `src/pymediaparser/vlm/__init__.py` | 子包入口 |
| **新建** | `src/pymediaparser/vlm/configs.py` | LocalVLMConfig, APIVLMConfig |
| **新建** | `src/pymediaparser/vlm/factory.py` | 字符串注册表 + create_vlm_client |
| **新建** | `src/pymediaparser/vlm/_local_base.py` | 本地模型公共基类 |
| **迁移** | `src/pymediaparser/vlm/qwen2.py` | Qwen2VLClient（重构为继承 _LocalTransformersBase） |
| **新建** | `src/pymediaparser/vlm/qwen3.py` | Qwen3VLClient |
| **新建** | `src/pymediaparser/vlm/openai_api.py` | OpenAIAPIClient |
| **改写** | `src/pymediaparser/vlm_qwen2.py` | 1 行 re-export |
| **修改** | `src/pymediaparser/__init__.py` | 更新 __all__ + __getattr__ 映射 |
| **修改** | `src/pymediaparser/vlm_base.py` | 环境变量名通用化 |
| **修改** | `src/pymediaparser/live_pipeline.py` | 删除硬编码 import + main() 用工厂 |
| **修改** | `scripts/run_parser.py` | 新增 --vlm-backend 等参数 |
| **修改** | `pyproject.toml` | 新增 vlm-qwen3 optional dep |
| **新建** | `tests/test_vlm_factory.py` | 工厂 + 配置类测试 |
| **新建** | `tests/test_vlm_openai_api.py` | API 客户端测试（mock HTTP） |

---

## 验证计划

### 1. 单元测试（无需 GPU）
```bash
PYTHON=/apprun/jiankai/python_test/conda_env/bin/python

# 配置类 + 工厂函数
$PYTHON -m pytest tests/test_vlm_factory.py -v

# API 客户端（mock HTTP）
$PYTHON -m pytest tests/test_vlm_openai_api.py -v

# 已有测试不回归
$PYTHON -m pytest tests/ -v
```

### 2. 向后兼容验证
```bash
$PYTHON -c "
from pymediaparser import VLMConfig, Qwen2VLClient, LivePipeline
from pymediaparser.vlm_qwen2 import Qwen2VLClient  # re-export
config = VLMConfig(device='cpu')
client = Qwen2VLClient(config)
print('向后兼容: OK')
"
```

### 3. 新导入路径
```bash
$PYTHON -c "
from pymediaparser.vlm import create_vlm_client, LocalVLMConfig, APIVLMConfig
from pymediaparser.vlm.factory import list_backends
print('可用后端:', list_backends())
print('新路径: OK')
"
```

### 4. Qwen3-VL 端到端推理（GPU，模型已在本地）
```bash
$PYTHON -c "
from pymediaparser import create_vlm_client, LocalVLMConfig
from PIL import Image

config = LocalVLMConfig(
    model_path='models/Qwen/Qwen3-VL-2B-Instruct',
    device='cuda:0', dtype='float16'
)
with create_vlm_client('qwen3-vl', config) as client:
    img = Image.open('resource/IMG_20260108_113053_HC.jpeg')
    result = client.analyze(img, '请描述画面内容')
    print(result.text[:200])
    print(f'推理耗时: {result.inference_time:.2f}s')
"
```

### 5. Qwen2-VL 通过工厂创建（确认重构无回归）
```bash
$PYTHON -c "
from pymediaparser import create_vlm_client, VLMConfig
from PIL import Image

config = VLMConfig(device='cuda:0')
with create_vlm_client('qwen2-vl', config) as client:
    img = Image.open('resource/IMG_20260108_113053_HC.jpeg')
    result = client.analyze(img, '请描述画面内容')
    print(result.text[:200])
"
```

### 6. CLI 验证
```bash
# 帮助信息
$PYTHON scripts/run_parser.py --help

# 新后端参数（API 模式，不实际连接也能验证参数解析）
$PYTHON scripts/run_parser.py \
    --url rtmp://host/stream \
    --vlm-backend openai-api \
    --api-base-url http://localhost:8000 \
    --api-model Qwen2-VL-7B-Instruct \
    --help
```
