"""VLM 子包：多后端视觉语言模型推理。

支持的后端：
- qwen2: Qwen2-VL 本地 transformers 推理
- qwen3: Qwen3-VL 本地 transformers 推理
- openai_api: OpenAI 兼容 API（vLLM / Ollama / TGI / OpenAI / 通义千问）

典型用法::

    from pymediaparser.vlm import create_vlm_client, LocalVLMConfig, APIVLMConfig

    # 本地模型
    client = create_vlm_client("qwen2", LocalVLMConfig(model_path="/path/to/model"))

    # API 服务
    client = create_vlm_client("openai_api", APIVLMConfig(base_url="http://localhost:8000/v1"))
"""

from .configs import APIVLMConfig, LocalVLMConfig
from .factory import create_vlm_client, list_backends, register_vlm_backend

__all__ = [
    "LocalVLMConfig",
    "APIVLMConfig",
    "create_vlm_client",
    "register_vlm_backend",
    "list_backends",
]
