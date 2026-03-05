"""
Qwen2-VL 视觉语言模型 GPU 版本
支持单张/多张图片输入，进行视觉问答
适用于远程 GPU 服务器环境
"""

import os
import sys
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 模型路径（可通过环境变量覆盖）
MODEL_PATH = os.environ.get(
    "QWEN_VL_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen2-VL-2B-Instruct")
)


class VLMAssistant:
    """视觉语言模型助手"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        初始化 VLM 助手
        
        Args:
            model_path: 模型路径，默认使用 MODEL_PATH
            device: 设备类型，"auto"/"cuda"/"cpu"
        """
        self.model_path = model_path or MODEL_PATH
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        
    def _get_device(self, device: str) -> str:
        """确定运行设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self, use_flash_attn: bool = True, max_pixels: int = 512*28*28):
        """
        加载模型
        
        Args:
            use_flash_attn: 是否使用 Flash Attention（需要 GPU）
            max_pixels: 最大图片像素数（用于控制显存占用）
        """
        print(f"正在加载模型: {self.model_path}")
        print(f"运行设备: {self.device}")
        
        # 根据设备选择数据类型 - T4等显存较小的GPU使用float16
        if self.device == "cuda":
            dtype = torch.float16  # float16更省显存
            print("使用 float16 精度（GPU优化）")
        else:
            dtype = torch.float32
            print("使用 float32 精度（CPU模式）")
        
        # 加载模型
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.device if self.device == "cuda" else None,
            "low_cpu_mem_usage": True,
        }
        
        # GPU 环境尝试使用 Flash Attention
        if self.device == "cuda" and use_flash_attn:
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("尝试启用 Flash Attention 2...")
            except:
                pass
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs
            )
        except Exception as e:
            # 如果 Flash Attention 失败，回退到普通加载
            if "flash" in str(e).lower():
                print("Flash Attention 不可用，使用默认注意力机制")
                load_kwargs.pop("attn_implementation", None)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )
            else:
                raise e
        
        # CPU 模式需要手动移动模型
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        self.model.eval()
        
        # 限制图片分辨率以节省显存（针对T4等显存较小的GPU）
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=256*28*28,
            max_pixels=max_pixels
        )
        
        # 显示 GPU 信息
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            if gpu_memory < 20:
                print(f"显存较小，已限制图片分辨率以避免OOM")
        
        print("模型加载完成！\n")
        return self
    
    def chat(
        self,
        images: list,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        进行视觉问答
        
        Args:
            images: 图片路径列表（支持本地路径和 URL）
            question: 问题文本
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            do_sample: 是否采样
            
        Returns:
            模型回答文本
        """
        if self.model is None:
            raise RuntimeError("请先调用 load_model() 加载模型")
        
        # 确保 images 是列表
        if isinstance(images, str):
            images = [images]
        
        # 构建消息
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        
        messages = [{"role": "user", "content": content}]
        
        # 准备输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        
        # 生成回复
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                use_cache=True
            )
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]


def demo_single_image(assistant: VLMAssistant):
    """单图分析演示"""
    print("=" * 60)
    print("  单图分析演示")
    print("=" * 60)
    
    # 使用网络图片
    image_url = "/apprun/jiankai/python_test/resource/IMG_20260108_113053_HC.jpeg"

    print(f"图片: {image_url}\n")
    
    question = "请详细描述这张图片中的内容，包括人物、场景和活动。"
    print(f"问题: {question}\n")
    
    print("正在分析...")
    answer = assistant.chat(image_url, question)
    print(f"\n回答: {answer}")


def demo_multi_image(assistant: VLMAssistant):
    """多图分析演示"""
    print("=" * 60)
    print("  多图分析演示")
    print("=" * 60)
    
    image_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaini.jpg"
    ]
    
    print("图片列表:")
    for i, url in enumerate(image_urls, 1):
        print(f"  {i}. {url}")
    print()
    
    question = "请分析这两张图片，比较它们的异同，并描述每张图片的主要内容。"
    print(f"问题: {question}\n")
    
    print("正在分析...")
    answer = assistant.chat(image_urls, question)
    print(f"\n回答: {answer}")


def interactive_mode(assistant: VLMAssistant):
    """交互模式"""
    print("=" * 60)
    print("  VLM 交互问答模式")
    print("  输入 'quit' 或 'q' 退出")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 40)
        image_input = input("图片路径/URL（多张用逗号分隔）: ").strip()
        
        if image_input.lower() in ['quit', 'exit', 'q', '']:
            if image_input == '':
                continue
            print("再见！")
            break
        
        # 解析图片路径
        images = [p.strip() for p in image_input.split(",")]
        
        # 验证本地文件
        valid_images = []
        for img in images:
            if img.startswith(("http://", "https://")):
                valid_images.append(img)
            elif os.path.exists(img):
                valid_images.append(img)
            else:
                print(f"警告: 文件不存在 - {img}")
        
        if not valid_images:
            print("没有有效的图片，请重新输入")
            continue
        
        question = input("问题: ").strip()
        if not question:
            question = "请描述这张图片的内容"
        
        print("\n正在分析...")
        try:
            answer = assistant.chat(valid_images, question)
            print(f"\n回答: {answer}")
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("   Qwen2-VL 视觉语言模型 GPU 版本")
    print("   支持单张/多张图片问答")
    print("=" * 60)
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"\nGPU 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("\n警告: CUDA 不可用，将使用 CPU 模式（速度较慢）")
    
    # 加载模型
    print("\n正在初始化模型...")
    assistant = VLMAssistant()
    assistant.load_model()
    
    # 菜单选择
    while True:
        print("\n请选择模式:")
        print("1. 单图分析演示（网络图片）")
        print("2. 多图分析演示（网络图片）")
        print("3. 交互问答模式")
        print("4. 退出")
        print()
        
        choice = input("请输入选项 (1-4): ").strip()
        
        if choice == "1":
            demo_single_image(assistant)
        elif choice == "2":
            demo_multi_image(assistant)
        elif choice == "3":
            interactive_mode(assistant)
        elif choice == "4":
            print("再见！")
            break
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()
