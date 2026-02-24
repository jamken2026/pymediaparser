"""结果输出处理器：将每帧 VLM 分析结果输出到不同目标。

当前实现：
- ConsoleResultHandler: 打印到控制台（默认）

未来可扩展：
- HttpCallbackHandler: HTTP POST 回调
- WebSocketHandler: WebSocket 推送

典型用法::

    from pymediaparser.result_handler import ConsoleResultHandler

    handler = ConsoleResultHandler()
    handler.handle(frame_result)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from .vlm_base import FrameResult

logger = logging.getLogger(__name__)


class ResultHandler(ABC):
    """结果处理器抽象基类。

    所有结果输出通道（控制台、HTTP、WebSocket 等）
    都需要继承此类并实现 ``handle`` 方法。
    """

    @abstractmethod
    def handle(self, result: FrameResult) -> None:
        """处理单个帧分析结果。

        Args:
            result: 包含帧时间戳、VLM 输出等信息的结果对象。
        """

    def on_start(self) -> None:
        """Pipeline 启动时的回调（可选覆写）。"""

    def on_stop(self) -> None:
        """Pipeline 停止时的回调（可选覆写）。"""


class ConsoleResultHandler(ResultHandler):
    """将分析结果打印到控制台。

    输出格式::

        [2026-02-09 14:30:01] Frame #0 | ts=1.000s | 耗时=0.82s
        >>> 画面中有一个人正在走路...
        ──────────────────────────────────────────

    Args:
        verbose: 是否显示详细信息（tokens 数、推理耗时等）。
    """

    SEPARATOR = "─" * 60

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def handle(self, result: FrameResult) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vlm = result.vlm_result

        # 头部行
        header = f"[{now}] Frame #{result.frame_index} | ts={result.timestamp:.3f}s"
        if self.verbose:
            header += f" | 推理耗时={vlm.inference_time:.2f}s | tokens={vlm.tokens_used}"
        print(header)

        # VLM 输出文本
        print(f">>> {vlm.text}")

        # 分隔线
        print(self.SEPARATOR)

    def on_start(self) -> None:
        print("\n" + "=" * 60)
        print("  实时流 VLM 分析 — 开始")
        print("=" * 60 + "\n")

    def on_stop(self) -> None:
        print("\n" + "=" * 60)
        print("  实时流 VLM 分析 — 结束")
        print("=" * 60 + "\n")


class HttpCallbackHandler(ResultHandler):
    """通过 HTTP POST 将结果推送到回调地址（预留实现）。

    Args:
        callback_url: HTTP 回调地址。
        timeout: 请求超时秒数。
    """

    def __init__(self, callback_url: str, timeout: float = 5.0) -> None:
        self.callback_url = callback_url
        self.timeout = timeout

    def _result_to_dict(self, result: FrameResult) -> dict[str, Any]:
        """将 FrameResult 序列化为可 JSON 化的字典。"""
        return {
            "frame_index": result.frame_index,
            "timestamp": result.timestamp,
            "capture_time": result.capture_time,
            "vlm_text": result.vlm_result.text,
            "inference_time": result.vlm_result.inference_time,
            "tokens_used": result.vlm_result.tokens_used,
        }

    def handle(self, result: FrameResult) -> None:
        import requests  # 延迟导入，避免未安装 requests 时报错

        payload = self._result_to_dict(result)
        try:
            resp = requests.post(
                self.callback_url,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.debug("HTTP 回调成功: %s -> %d", self.callback_url, resp.status_code)
        except Exception as exc:
            logger.error("HTTP 回调失败: %s — %s", self.callback_url, exc)
