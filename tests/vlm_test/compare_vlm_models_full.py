"""
Qwen3.5-2B vs Qwen3-VL-2B 全面对比测试
测试项目：
1. 单图理解 - 简单图形
2. 单图理解 - 复杂场景
3. 多图理解 - 图片对比
4. OCR能力 - 文字识别
5. 视频理解（可选）
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

# 测试资源
TEST_RESOURCES = {
    "simple_image": "/apprun/jiankai/python_test/resource/test_img1.png",
    "complex_image": "/apprun/jiankai/python_test/resource/test_img2.png",
    "video": "/apprun/jiankai/python_test/resource/866f893bb7a353c75f1aa5c7cb61e4e3.mp4",
    "ocr_image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
}

# 网络测试图片
WEB_IMAGES = {
    "scene1": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    "scene2": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaini.jpg",
}


def load_model(model_name, model_config):
    """加载模型"""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
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
    
    memory_gb = 0
    if device == "cuda":
        memory_gb = torch.cuda.memory_allocated(0) / (1024**3)
    
    return {
        "model": model,
        "processor": processor,
        "load_time": load_time,
        "memory_gb": memory_gb,
    }


def run_inference(model_info, messages, max_new_tokens=256):
    """运行推理"""
    model = model_info["model"]
    processor = model_info["processor"]
    
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    inference_time = time.time() - start_time
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return {
        "output": output_text,
        "inference_time": inference_time,
    }


def test_simple_image(model_info, model_name):
    """测试1: 简单图形理解"""
    print(f"\n[测试1] 简单图形理解 - {model_name}")
    print("-" * 50)
    
    test_image = TEST_RESOURCES["simple_image"]
    if not os.path.exists(test_image):
        test_image = WEB_IMAGES["scene1"]
    
    print(f"图片: {test_image}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "请描述这张图片的内容。"},
            ],
        }
    ]
    
    result = run_inference(model_info, messages)
    
    print(f"推理时间: {result['inference_time']:.2f}s")
    print(f"回答: {result['output'][:200]}...")
    
    return result


def test_complex_scene(model_info, model_name):
    """测试2: 复杂场景理解"""
    print(f"\n[测试2] 复杂场景理解 - {model_name}")
    print("-" * 50)
    
    test_image = TEST_RESOURCES["complex_image"]
    if not os.path.exists(test_image):
        test_image = WEB_IMAGES["scene2"]
    
    print(f"图片: {test_image}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "请详细描述这张图片中的场景、人物和活动，尽可能详细地说明你看到的细节。"},
            ],
        }
    ]
    
    result = run_inference(model_info, messages, max_new_tokens=512)
    
    print(f"推理时间: {result['inference_time']:.2f}s")
    print(f"回答: {result['output'][:300]}...")
    
    return result


def test_multi_image(model_info, model_name):
    """测试3: 多图对比理解"""
    print(f"\n[测试3] 多图对比理解 - {model_name}")
    print("-" * 50)
    
    print(f"图片: 2张网络图片")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": WEB_IMAGES["scene1"]},
                {"type": "image", "image": WEB_IMAGES["scene2"]},
                {"type": "text", "text": "请比较这两张图片的异同点，包括场景、人物、活动等方面。"},
            ],
        }
    ]
    
    result = run_inference(model_info, messages, max_new_tokens=512)
    
    print(f"推理时间: {result['inference_time']:.2f}s")
    print(f"回答: {result['output'][:300]}...")
    
    return result


def test_ocr(model_info, model_name):
    """测试4: OCR文字识别"""
    print(f"\n[测试4] OCR文字识别 - {model_name}")
    print("-" * 50)
    
    print(f"图片: {WEB_IMAGES['scene1']}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": WEB_IMAGES["scene1"]},
                {"type": "text", "text": "请识别图片中的所有文字内容，并按从上到下、从左到右的顺序列出。"},
            ],
        }
    ]
    
    result = run_inference(model_info, messages, max_new_tokens=256)
    
    print(f"推理时间: {result['inference_time']:.2f}s")
    print(f"回答: {result['output'][:200]}...")
    
    return result


def test_video(model_info, model_name):
    """测试5: 视频理解"""
    print(f"\n[测试5] 视频理解 - {model_name}")
    print("-" * 50)
    
    video_path = TEST_RESOURCES["video"]
    
    if not os.path.exists(video_path):
        print("本地视频不存在，使用网络视频")
        video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
        video_source = video_url
    else:
        video_source = video_path
    
    print(f"视频: {video_source}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_source, "fps": 1.0},
                {"type": "text", "text": "请描述这个视频的内容，包括场景变化和主要活动。"},
            ],
        }
    ]
    
    try:
        result = run_inference(model_info, messages, max_new_tokens=256)
        print(f"推理时间: {result['inference_time']:.2f}s")
        print(f"回答: {result['output'][:200]}...")
        return result
    except Exception as e:
        print(f"视频测试失败: {e}")
        return {"error": str(e)}


def test_reasoning(model_info, model_name):
    """测试6: 逻辑推理能力"""
    print(f"\n[测试6] 逻辑推理能力 - {model_name}")
    print("-" * 50)
    
    # 使用一张图片进行推理测试
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": WEB_IMAGES["scene1"]},
                {"type": "text", "text": "观察这张图片，分析：1. 这是什么场景？2. 图片中人物的可能的职业是什么？3. 推测他们正在做什么？请给出你的推理过程。"},
            ],
        }
    ]
    
    result = run_inference(model_info, messages, max_new_tokens=512)
    
    print(f"推理时间: {result['inference_time']:.2f}s")
    print(f"回答: {result['output'][:300]}...")
    
    return result


def print_full_comparison(results):
    """打印完整对比结果"""
    print("\n" + "=" * 70)
    print("  完整对比结果汇总")
    print("=" * 70)
    
    print("\n### 模型基本信息")
    print("-" * 50)
    for model_name, data in results.items():
        print(f"{model_name}:")
        print(f"  加载时间: {data['load_time']:.2f}s")
        print(f"  显存占用: {data['memory_gb']:.2f} GB")
    
    tests = ["简单图形", "复杂场景", "多图对比", "OCR识别", "逻辑推理"]
    test_keys = ["simple", "complex", "multi", "ocr", "reasoning"]
    
    print("\n### 推理时间对比 (秒)")
    print("-" * 50)
    print(f"{'测试项目':<15} {'Qwen3-VL-2B':>15} {'Qwen3.5-2B':>15}")
    print("-" * 50)
    
    for test_name, test_key in zip(tests, test_keys):
        times = []
        for model_name in MODELS.keys():
            if test_key in results[model_name] and "inference_time" in results[model_name][test_key]:
                times.append(f"{results[model_name][test_key]['inference_time']:.2f}")
            else:
                times.append("N/A")
        print(f"{test_name:<15} {times[0]:>15} {times[1]:>15}")
    
    print("\n### 回答质量对比 (字数)")
    print("-" * 50)
    print(f"{'测试项目':<15} {'Qwen3-VL-2B':>15} {'Qwen3.5-2B':>15}")
    print("-" * 50)
    
    for test_name, test_key in zip(tests, test_keys):
        lengths = []
        for model_name in MODELS.keys():
            if test_key in results[model_name] and "output" in results[model_name][test_key]:
                lengths.append(str(len(results[model_name][test_key]["output"])))
            else:
                lengths.append("N/A")
        print(f"{test_name:<15} {lengths[0]:>15} {lengths[1]:>15}")


def main():
    print("\n" + "=" * 70)
    print("   Qwen3.5-2B vs Qwen3-VL-2B 全面对比测试")
    print("=" * 70)
    
    # GPU 信息
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # 检查模型
    for model_name, config in MODELS.items():
        if not os.path.exists(config["path"]):
            print(f"模型不存在: {config['path']}")
            return
    
    results = {}
    
    # 逐个测试模型
    for model_name, config in MODELS.items():
        print(f"\n\n{'#' * 70}")
        print(f"# 测试模型: {model_name}")
        print("#" * 70)
        
        # 加载模型
        model_info = load_model(model_name, config)
        results[model_name] = {
            "load_time": model_info["load_time"],
            "memory_gb": model_info["memory_gb"],
        }
        
        print(f"加载完成: {model_info['load_time']:.2f}s, 显存: {model_info['memory_gb']:.2f} GB")
        
        # 运行测试
        try:
            results[model_name]["simple"] = test_simple_image(model_info, model_name)
        except Exception as e:
            print(f"简单图形测试失败: {e}")
            results[model_name]["simple"] = {"error": str(e)}
        
        try:
            results[model_name]["complex"] = test_complex_scene(model_info, model_name)
        except Exception as e:
            print(f"复杂场景测试失败: {e}")
            results[model_name]["complex"] = {"error": str(e)}
        
        try:
            results[model_name]["multi"] = test_multi_image(model_info, model_name)
        except Exception as e:
            print(f"多图对比测试失败: {e}")
            results[model_name]["multi"] = {"error": str(e)}
        
        try:
            results[model_name]["ocr"] = test_ocr(model_info, model_name)
        except Exception as e:
            print(f"OCR测试失败: {e}")
            results[model_name]["ocr"] = {"error": str(e)}
        
        try:
            results[model_name]["reasoning"] = test_reasoning(model_info, model_name)
        except Exception as e:
            print(f"推理测试失败: {e}")
            results[model_name]["reasoning"] = {"error": str(e)}
        
        # 清理显存
        del model_info["model"]
        del model_info["processor"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n已释放 {model_name} 显存")
    
    # 打印对比结果
    print_full_comparison(results)
    
    # 打印详细回答
    print("\n" + "=" * 70)
    print("  详细回答内容")
    print("=" * 70)
    
    for test_name, test_key in [("简单图形", "simple"), ("复杂场景", "complex"), ("逻辑推理", "reasoning")]:
        print(f"\n### {test_name}测试回答对比")
        print("-" * 50)
        for model_name in MODELS.keys():
            if test_key in results[model_name] and "output" in results[model_name][test_key]:
                output = results[model_name][test_key]["output"]
                print(f"\n[{model_name}]:")
                print(f"{output[:400]}{'...' if len(output) > 400 else ''}")
    
    print("\n\n" + "=" * 70)
    print("  测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
