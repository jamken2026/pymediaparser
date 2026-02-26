"""VLM 工厂函数、配置类、后端注册的单元测试。

不依赖 transformers / torch 等重型库，通过 mock 验证逻辑。
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from pymediaparser.vlm_base import VLMConfig, VLMResult, VLMClient
from pymediaparser.vlm.configs import LocalVLMConfig, APIVLMConfig
from pymediaparser.vlm.factory import (
    create_vlm_client,
    register_vlm_backend,
    list_backends,
    _REGISTRY,
)


# =====================================================================
# 配置类测试
# =====================================================================

class TestLocalVLMConfig:
    """LocalVLMConfig 继承 VLMConfig 的测试。"""

    def test_inherits_vlm_config(self):
        """LocalVLMConfig 是 VLMConfig 的子类。"""
        cfg = LocalVLMConfig()
        assert isinstance(cfg, VLMConfig)

    def test_default_values_same_as_vlmconfig(self):
        """默认值与 VLMConfig 一致。"""
        base = VLMConfig()
        local = LocalVLMConfig()
        assert local.device == base.device
        assert local.dtype == base.dtype
        assert local.max_new_tokens == base.max_new_tokens
        assert local.default_prompt == base.default_prompt

    def test_custom_values(self):
        """可自定义字段值。"""
        cfg = LocalVLMConfig(
            model_path="/my/model",
            device="cpu",
            dtype="bfloat16",
            max_new_tokens=512,
        )
        assert cfg.model_path == "/my/model"
        assert cfg.device == "cpu"
        assert cfg.dtype == "bfloat16"
        assert cfg.max_new_tokens == 512


class TestAPIVLMConfig:
    """APIVLMConfig 独立配置的测试。"""

    def test_default_values(self):
        """验证默认值。"""
        cfg = APIVLMConfig()
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.api_key is None
        assert cfg.model_name == "default"
        assert cfg.timeout == 60.0
        assert cfg.max_retries == 3
        assert cfg.image_max_size == 1024
        assert cfg.image_quality == 85
        assert cfg.max_new_tokens == 256

    def test_not_vlmconfig_subclass(self):
        """APIVLMConfig 不是 VLMConfig 的子类。"""
        cfg = APIVLMConfig()
        assert not isinstance(cfg, VLMConfig)

    def test_custom_values(self):
        """可自定义字段值。"""
        cfg = APIVLMConfig(
            base_url="http://my-server:9000/v1",
            api_key="sk-test-123",
            model_name="qwen2-vl",
            timeout=30.0,
            max_retries=5,
        )
        assert cfg.base_url == "http://my-server:9000/v1"
        assert cfg.api_key == "sk-test-123"
        assert cfg.model_name == "qwen2-vl"
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 5


# =====================================================================
# 工厂函数测试
# =====================================================================

class TestListBackends:
    """list_backends() 测试。"""

    def test_contains_builtin_backends(self):
        """内置后端包含 qwen2, qwen3, openai_api。"""
        backends = list_backends()
        assert "qwen2" in backends
        assert "qwen3" in backends
        assert "openai_api" in backends

    def test_returns_list(self):
        """返回列表类型。"""
        assert isinstance(list_backends(), list)


class TestCreateVLMClient:
    """create_vlm_client() 测试。"""

    def test_unknown_backend_raises_valueerror(self):
        """未知后端抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知的 VLM 后端"):
            create_vlm_client("nonexistent_backend")

    def test_error_message_shows_available_backends(self):
        """错误信息包含可用后端列表。"""
        with pytest.raises(ValueError, match="openai_api"):
            create_vlm_client("bad_name")

    @patch("pymediaparser.vlm.factory.importlib.import_module")
    def test_creates_client_with_config(self, mock_import):
        """正确传递 config 创建客户端。"""
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.Qwen2VLClient = mock_cls
        mock_import.return_value = mock_module

        config = VLMConfig(device="cpu")
        create_vlm_client("qwen2", config)

        mock_import.assert_called_once_with("pymediaparser.vlm.qwen2")
        mock_cls.assert_called_once_with(config)

    @patch("pymediaparser.vlm.factory.importlib.import_module")
    def test_creates_client_without_config(self, mock_import):
        """不传 config 时使用默认值创建。"""
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.OpenAIAPIClient = mock_cls
        mock_import.return_value = mock_module

        create_vlm_client("openai_api")

        mock_cls.assert_called_once_with()

    def test_import_error_gives_helpful_message(self):
        """模块导入失败时给出有用提示。"""
        # 注册一个不存在的模块
        _REGISTRY["_test_bad"] = ("nonexistent.module.path", "BadClass")
        try:
            with pytest.raises(ImportError, match="无法导入后端"):
                create_vlm_client("_test_bad")
        finally:
            del _REGISTRY["_test_bad"]


class TestRegisterVLMBackend:
    """register_vlm_backend() 测试。"""

    def test_register_new_backend(self):
        """可以注册新后端。"""
        name = "_test_custom_vlm"
        try:
            register_vlm_backend(name, "my_pkg.my_vlm", "MyVLMClient")
            assert name in _REGISTRY
            assert _REGISTRY[name] == ("my_pkg.my_vlm", "MyVLMClient")
        finally:
            _REGISTRY.pop(name, None)

    def test_duplicate_registration_raises(self):
        """重复注册同名后端抛出 ValueError。"""
        name = "_test_dup"
        _REGISTRY[name] = ("mod", "Cls")
        try:
            with pytest.raises(ValueError, match="已注册"):
                register_vlm_backend(name, "other_mod", "OtherCls")
        finally:
            del _REGISTRY[name]

    def test_registered_backend_appears_in_list(self):
        """注册后出现在 list_backends 中。"""
        name = "_test_listed"
        try:
            register_vlm_backend(name, "mod", "Cls")
            assert name in list_backends()
        finally:
            _REGISTRY.pop(name, None)


# =====================================================================
# 延迟导入映射测试（通过 pymediaparser 包级导入）
# =====================================================================

class TestPackageLevelImports:
    """验证通过包级 __getattr__ 延迟导入的正确性。"""

    def test_import_vlm_config(self):
        """from pymediaparser import VLMConfig 可用。"""
        from pymediaparser import VLMConfig as V
        assert V is VLMConfig

    def test_import_local_vlm_config(self):
        """from pymediaparser import LocalVLMConfig 可用。"""
        from pymediaparser import LocalVLMConfig as L
        assert L is LocalVLMConfig

    def test_import_api_vlm_config(self):
        """from pymediaparser import APIVLMConfig 可用。"""
        from pymediaparser import APIVLMConfig as A
        assert A is APIVLMConfig

    def test_import_create_vlm_client(self):
        """from pymediaparser import create_vlm_client 可用。"""
        from pymediaparser import create_vlm_client as f
        assert callable(f)

    def test_import_list_backends(self):
        """from pymediaparser import list_backends 可用。"""
        from pymediaparser import list_backends as lb
        assert callable(lb)

    def test_import_register_vlm_backend(self):
        """from pymediaparser import register_vlm_backend 可用。"""
        from pymediaparser import register_vlm_backend as r
        assert callable(r)

    def test_nonexistent_attr_raises(self):
        """访问不存在的属性抛出 ImportError（由 __getattr__ 的 AttributeError 转换）。"""
        with pytest.raises(ImportError):
            from pymediaparser import NonExistentThing  # noqa: F401
