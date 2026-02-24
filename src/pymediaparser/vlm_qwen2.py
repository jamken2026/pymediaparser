"""Qwen2-VL 视觉语言模型推理客户端。

基于 ``transformers`` + ``qwen-vl-utils`` 的本地推理实现，
继承 :class:`VLMClient` 抽象接口。

典型用法::

    from pymediaparser.vlm_base import VLMConfig
    from pymediaparser.vlm_qwen2 import Qwen2VLClient

    config = VLMConfig(device="cuda:0")
    with Qwen2VLClient(config) as client:
        result = client.analyze(pil_image, "请描述画面内容。")
        print(result.text)
"""

from __future__ import annotations

import logging
import time
from typing import Any, List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from .vlm_base import VLMClient, VLMConfig, VLMResult

logger = logging.getLogger(__name__)

# torch dtype 映射
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class Qwen2VLClient(VLMClient):
    """Qwen2-VL 本地推理客户端。

    在 T4 GPU 上使用 float16 精度，Qwen2-VL-2B-Instruct 模型
    约占 4~5 GB 显存，单帧推理耗时约 0.5~1.5 秒。

    Args:
        config: VLM 配置（模型路径、设备、精度等）。
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self._model: Qwen2VLForConditionalGeneration | None = None
        self._processor: Any = None  # 运行时为 Qwen2VLProcessor，含 apply_chat_template 等动态方法

    # ------------------------------------------------------------------
    # VLMClient 接口实现
    # ------------------------------------------------------------------

    def load(self) -> None:
        """加载 Qwen2-VL 模型和处理器到指定设备。"""
        cfg = self.config
        dtype = _DTYPE_MAP.get(cfg.dtype, torch.float16)

        logger.info(
            "加载模型: %s  device=%s  dtype=%s",
            cfg.model_path, cfg.device, cfg.dtype,
        )

        # ---- 加载模型 ------------------------------------------------
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        # 设备映射
        if "cuda" in cfg.device:
            load_kwargs["device_map"] = cfg.device
        # else: 加载到 CPU，后续手动移动

        # Flash Attention 2
        if cfg.use_flash_attn and "cuda" in cfg.device:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("尝试启用 Flash Attention 2 ...")

        try:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                cfg.model_path, **load_kwargs
            )
        except Exception as exc:
            # Flash Attention 不可用时回退
            if "flash" in str(exc).lower() and "attn_implementation" in load_kwargs:
                logger.warning("Flash Attention 2 不可用，回退到默认注意力机制")
                load_kwargs.pop("attn_implementation", None)
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    cfg.model_path, **load_kwargs
                )
            else:
                raise

        # CPU 模式手动移动
        if "cuda" not in cfg.device:
            assert self._model is not None
            self._model = self._model.to(torch.device(cfg.device))  # pyright: ignore[reportArgumentType]

        self._model.eval()

        # ---- 加载处理器 ----------------------------------------------
        self._processor = AutoProcessor.from_pretrained(
            cfg.model_path,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
        )

        # ---- 打印 GPU 信息 -------------------------------------------
        if "cuda" in cfg.device and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

        logger.info("模型加载完成")

    def analyze(self, image: Image.Image, prompt: str | None = None) -> VLMResult:
        """对单帧图像进行 VLM 推理。

        Args:
            image: RGB 格式 PIL 图像。
            prompt: 文本提示；为 None 时使用 ``config.default_prompt``。

        Returns:
            包含文本回复和推理统计的 VLMResult。

        Raises:
            RuntimeError: 模型未加载时调用。
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        prompt = prompt or self.config.default_prompt
        t0 = time.perf_counter()

        # ---- 构造多模态消息 ------------------------------------------
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # ---- 处理输入 ------------------------------------------------
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore[assignment]  # 库的类型注解声明返回3元组，但 return_video_kwargs=False 时实际返回2元组

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 移动到模型所在设备
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)

        # ---- 推理 ----------------------------------------------------
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # 确定性输出（实时分析场景更稳定）
                use_cache=True,
            )

        # ---- 解码输出 ------------------------------------------------
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elapsed = time.perf_counter() - t0
        tokens_used = generated_ids_trimmed[0].shape[0] if generated_ids_trimmed else 0

        logger.debug(
            "推理完成: %.2fs, %d tokens — %s",
            elapsed, tokens_used, output_text[:80],
        )

        return VLMResult(
            text=output_text.strip(),
            tokens_used=int(tokens_used),
            inference_time=elapsed,
        )

    def unload(self) -> None:
        """释放模型，清空 GPU 显存。"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("模型已卸载，显存已释放")

    # ------------------------------------------------------------------
    # 批量处理功能（从 EnhancedVLMClient 合并）
    # ------------------------------------------------------------------

    def analyze_batch(self, images: List[Image.Image], prompt: str) -> VLMResult:
        """对多张图像进行批量 VLM 推理。

        Args:
            images: PIL 图像列表。
            prompt: 文本提示词。

        Returns:
            包含批量分析结果的 VLMResult。
        """
        if not images:
            raise ValueError("图像列表不能为空")

        if self._model is None or self._processor is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        start_time = time.perf_counter()

        try:
            # 构造多图像消息
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            # 处理输入
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)  # type: ignore[assignment]  # 库的类型注解声明返回3元组，但 return_video_kwargs=False 时实际返回2元组

            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # 移动到设备
            device = next(self._model.parameters()).device
            inputs = inputs.to(device)

            # 推理
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=getattr(self.config, 'temperature', 0.7),
                    do_sample=getattr(self.config, 'do_sample', True),
                    use_cache=True,
                )

            # 解码输出
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            inference_time = time.perf_counter() - start_time

            input_tokens = len(inputs.input_ids[0])
            output_tokens = len(generated_ids_trimmed[0])

            logger.debug(
                "批量处理完成 - 图像数: %d, 输入tokens: %d, 输出tokens: %d, 耗时: %.2fs",
                len(images), input_tokens, output_tokens, inference_time
            )

            return VLMResult(
                text=output_text,
                inference_time=inference_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            logger.error("批量处理失败: %s", e, exc_info=True)
            raise

    def analyze_with_context(self, images: List[Image.Image],
                            context_prompt: str | None = None) -> VLMResult:
        """带上下文的批量分析。

        Args:
            images: PIL 图像列表。
            context_prompt: 上下文提示词，为 None 时自动生成。

        Returns:
            分析结果 VLMResult。
        """
        if context_prompt is None:
            context_prompt = self._build_context_prompt(len(images))
        return self.analyze_batch(images, context_prompt)

    def _build_context_prompt(self, image_count: int) -> str:
        """根据图像数量构建上下文提示词。"""
        if image_count == 1:
            return "请详细分析这张图像的内容。"
        elif image_count <= 3:
            return f"请分析这{image_count}张图像，描述它们之间的关系和变化。"
        else:
            return (
                f"请分析这{image_count}帧连续图像序列，"
                f"描述场景的发展变化和重要事件。"
            )
