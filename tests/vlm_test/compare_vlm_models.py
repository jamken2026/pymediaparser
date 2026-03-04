"""
Qwen3.5-2B vs Qwen3-VL-2B 模型对比测试脚本
"""

import os
import sys
import time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

MODELS = {
    "Qwen3-VL-2B": {
        "path": os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen3-VL-2B-Instruct"),
    },
    "Qwen3.5-2B": {
        "path": os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen3.5-2B"),
    },
}

TEST_IMAGE = "/apprun/jiankai/python_test/resource/test_img1.png"


def check_gpu():
    print("=" * 60)
    print("  GPU 状态")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"总显存: {gpu_memory:.2f} GB")
        return True
    else:
        print("CUDA 不可用")
        return False


def load_model(model_name, model_config):
    print(f"\n{'=' * 60}")
    print(f"  加载模型: {model_name}")
    print("=" * 60)
    
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"模型路径: {model_config['path']}")
    print(f"设备: {device}")
    
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device if device == "cuda" else None,
        "low_cpu_mem_usage": True,
    }
    
    if device == "cuda":
        load_kwargs["attn_implementation"] = "flash_attention_2"
    
    start_time = time.time()
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_config["path"], **load_kwargs
        )
    except Exception as e:
        if "flash" in str(e).lower():
            print("Flash Attention 不可用，使用默认注意力")
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(
                model_config["path"], **load_kwargs
            )
        else:
            raise
    
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_config["path"],
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    
    load_time = time.time() - start_time
    print(f"加载完成，耗时: {load_time:.2f}s")
    
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"显存占用: {allocated:.2f} GB")
    
    return {"model": model, "processor": processor, "load_time": load_time}


def test_single_image(model_info, model_name):
    print(f"\n{'=' * 60}")
    print(f"  [{model_name}] 单图理解测试")
    print("=" * 60)
    
    model = model_info["model"]
    processor = model_info["processor"]
    
    test_image = TEST_IMAGE
    if not os.path.exists(test_image):
        test_image = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    print(f"测试图片: {test_image}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "请详细描述这张图片的内容。"},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    inference_time = time.time() - start_time
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"\n推理耗时: {inference_time:.2f}s")
    print(f"\n回答:\n{output_text[:500]}...")
    
    return {"inference_time": inference_time, "output": output_text}


def print_comparison(results):
    print("\n" + "=" * 60)
    print("  对比结果汇总")
    print("=" * 60)
    
    print("\n### 加载时间对比")
    for model_name, data in results.items():
        print(f"{model_name}: {data['load_time']:.2f}s")
    
    print("\n### 推理时间对比（单图）")
    for model_name, data in results.items():
        if "single_image" in data and "inference_time" in data["single_image"]:
            print(f"{model_name}: {data['single_image']['inference_time']:.2f}s")


def main():
    print("\n" + "=" * 60)
    print("   Qwen3.5-2B vs Qwen3-VL-2B 对比测试")
    print("=" * 60)
    
    check_gpu()
    
    for model_name, config in MODELS.items():
        if not os.path.exists(config["path"]):
            print(f"模型不存在: {config['path']}")
            return
    
    results = {}
    
    for model_name, config in MODELS.items():
        print(f"\n\n{'#' * 60}")
        print(f"# 测试模型: {model_name}")
        print("#" * 60)
        
        model_info = load_model(model_name, config)
        results[model_name] = {"load_time": model_info["load_time"]}
        
        try:
            results[model_name]["single_image"] = test_single_image(model_info, model_name)
        except Exception as e:
            print(f"单图测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[model_name]["single_image"] = {"error": str(e)}
        
        del model_info["model"]
        del model_info["processor"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n已释放显存")
    
    print_comparison(results)
    print("\n测试完成！")


if __name__ == "__main__":
    main()
