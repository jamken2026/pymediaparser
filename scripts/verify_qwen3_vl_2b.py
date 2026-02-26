"""
Qwen3-VL-2B-Instruct 模型验证脚本

验证内容：
1. 模型加载
2. 单图理解
3. 多图理解
4. 视频理解（可选）
5. 显存占用监控
"""

import os
import sys
import time
import torch
from PIL import Image

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 模型配置
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_LOCAL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen3-VL-2B-Instruct")


def check_dependencies():
    """检查依赖版本"""
    print("=" * 60)
    print("  依赖检查")
    print("=" * 60)
    
    import transformers
    print(f"transformers 版本: {transformers.__version__}")
    
    # Qwen3-VL 需要 transformers >= 4.57.0
    version_parts = transformers.__version__.split(".")
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    
    if major < 4 or (major == 4 and minor < 57):
        print(f"⚠️  警告: Qwen3-VL 需要 transformers >= 4.57.0")
        print(f"   当前版本 {transformers.__version__} 可能不兼容")
        print("   请运行: pip install 'transformers>=4.57.0'")
        return False
    
    print("✅ transformers 版本符合要求")
    
    # 检查 qwen-vl-utils
    try:
        import qwen_vl_utils
        print(f"✅ qwen-vl-utils 已安装")
    except ImportError:
        print("⚠️  qwen-vl-utils 未安装")
        print("   请运行: pip install qwen-vl-utils")
        return False
    
    return True


def check_gpu():
    """检查 GPU 状态"""
    print("\n" + "=" * 60)
    print("  GPU 状态")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_free = torch.cuda.memory_reserved(0) / (1024**3)
        
        print(f"GPU: {gpu_name}")
        print(f"总显存: {gpu_memory:.2f} GB")
        print(f"已用显存: {gpu_free:.2f} GB")
        print(f"可用显存: {gpu_memory - gpu_free:.2f} GB")
        
        if gpu_memory < 6:
            print("⚠️  警告: 显存较小，可能需要使用量化版本")
        
        return True
    else:
        print("⚠️  CUDA 不可用，将使用 CPU 模式（速度较慢）")
        return False


def download_model():
    """下载模型"""
    print("\n" + "=" * 60)
    print("  模型下载")
    print("=" * 60)
    
    # 检查本地是否已存在
    if os.path.exists(MODEL_LOCAL_PATH):
        # 检查是否有必要文件
        required_files = ["config.json"]
        safetensors_file = os.path.join(MODEL_LOCAL_PATH, "model.safetensors")
        safetensors_index = os.path.join(MODEL_LOCAL_PATH, "model.safetensors.index.json")        
        # 至少要有 config.json 和一个模型文件
        if os.path.exists(os.path.join(MODEL_LOCAL_PATH, "config.json")) and \
           (os.path.exists(safetensors_file) or os.path.exists(safetensors_index)):
            print(f"✅ 模型已存在于: {MODEL_LOCAL_PATH}")
            return MODEL_LOCAL_PATH
    
    print(f"正在下载模型: {MODEL_ID}")
    print("这可能需要几分钟...")
    
    # 创建目标目录
    os.makedirs(MODEL_LOCAL_PATH, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        
        # 国内镜像列表
        mirrors = [
            {
                "name": "HuggingFace 镜像 (hf-mirror.com)",
                "endpoint": "https://hf-mirror.com",
                "env": "HF_ENDPOINT"
            },
            {
                "name": "ModelScope",
                "endpoint": None,
                "env": None,
                "use_modelscope": True
            },
            {
                "name": "HuggingFace 官方",
                "endpoint": None,
                "env": None
            }
        ]
        
        # 优先使用镜像
        use_modelscope = os.environ.get("USE_MODELSCOPE", "false").lower() == "true"
        use_hf_mirror = os.environ.get("USE_HF_MIRROR", "true").lower() == "true"
                
        if use_modelscope:
            # 使用 ModelScope
            print("使用 ModelScope 下载...")
            try:
                from modelscope import snapshot_download as ms_download
                model_path = ms_download(
                    "Qwen/Qwen3-VL-2B-Instruct",
                    cache_dir=os.path.join(PROJECT_ROOT, "models")
                )
                print(f"✅ 模型下载完成: {model_path}")
                return model_path
            except ImportError:
                print("⚠️  ModelScope 未安装，尝试其他镜像...")
        
        # 尝试 HuggingFace 镜像
        for mirror in mirrors:
            if mirror.get("use_modelscope"):
                continue  # 跳过 ModelScope（已处理）
            
            try:
                print(f"\n尝试: {mirror['name']}...")
                
                # 设置镜像环境变量
                old_endpoint = os.environ.get("HF_ENDPOINT")
                if mirror["endpoint"]:
                    os.environ["HF_ENDPOINT"] = mirror["endpoint"]
                    print(f"  镜像地址: {mirror['endpoint']}")
                
                model_path = snapshot_download(
                    MODEL_ID,
                    local_dir=MODEL_LOCAL_PATH,
                    etag_timeout=30,
                    resume_download=True
                )
                
                # 恢复环境变量
                if old_endpoint:
                    os.environ["HF_ENDPOINT"] = old_endpoint
                elif "HF_ENDPOINT" in os.environ:
                    del os.environ["HF_ENDPOINT"]
                
                print(f"✅ 模型下载完成: {model_path}")
                return model_path
                
            except Exception as e:
                print(f"  ❌ {mirror['name']} 失败: {e}")
                continue
        
        print("\n❌ 所有镜像都下载失败")
        return None
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print("\n备选方案:")
        print("1. 手动下载模型到 models/Qwen/Qwen3-VL-2B-Instruct/")
        print("2. 设置 USE_MODELSCOPE=true 使用 ModelScope")
        print("3. 设置 USE_HF_MIRROR=true 使用国内镜像")
        return None


def load_model(model_path: str):
    """加载模型"""
    print("\n" + "=" * 60)
    print("  模型加载")
    print("=" * 60)
    
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    print(f"精度: {dtype}")
    
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device if device == "cuda" else None,
        "low_cpu_mem_usage": True,
    }
    
    # 尝试 Flash Attention
    if device == "cuda":
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("尝试启用 Flash Attention 2...")
        except:
            pass
    
    start_time = time.time()
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        if "flash" in str(e).lower():
            print("Flash Attention 不可用，使用默认注意力机制")
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        else:
            raise
    
    if device == "cpu":
        model = model.to("cpu")
    
    model.eval()
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,  # 限制分辨率节省显存
    )
    
    load_time = time.time() - start_time
    print(f"✅ 模型加载完成，耗时: {load_time:.2f}s")
    
    # 显示显存占用
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"显存占用: {allocated:.2f} GB (已分配), {reserved:.2f} GB (已预留)")
    
    return model, processor


def test_single_image(model, processor):
    """测试单图理解"""
    print("\n" + "=" * 60)
    print("  测试 1: 单图理解")
    print("=" * 60)
    
    # 使用本地测试图片或网络图片
    test_image = "/apprun/jiankai/python_test/resource/IMG_20260108_113053_HC.jpeg"
    
    if not os.path.exists(test_image):
        test_image = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    print(f"测试图片: {test_image}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "请详细描述这张图片的内容，包括人物、场景和活动。"},
            ],
        }
    ]
    
    # 准备输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # 推理
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    inference_time = time.time() - start_time
    
    # 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"\n推理耗时: {inference_time:.2f}s")
    print(f"\n模型回答:\n{output_text}")
    
    return True


def test_multi_image(model, processor):
    """测试多图理解"""
    print("\n" + "=" * 60)
    print("  测试 2: 多图理解")
    print("=" * 60)
    
    # 使用网络图片
    test_images = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaini.jpg"
    ]
    
    print(f"测试图片数量: {len(test_images)}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_images[0]},
                {"type": "image", "image": test_images[1]},
                {"type": "text", "text": "请比较这两张图片的异同点。"},
            ],
        }
    ]
    
    # 准备输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # 推理
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    inference_time = time.time() - start_time
    
    # 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"\n推理耗时: {inference_time:.2f}s")
    print(f"\n模型回答:\n{output_text}")
    
    return True


def test_video(model, processor):
    """测试视频理解（可选）"""
    print("\n" + "=" * 60)
    print("  测试 3: 视频理解（可选）")
    print("=" * 60)
    
    # 使用网络视频
    test_video = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
    
    print(f"测试视频: {test_video}")
    print("注意: 视频理解需要更多显存和时间")
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": test_video, "fps": 1.0},  # 低帧率节省显存
                    {"type": "text", "text": "请描述这个视频的内容。"},
                ],
            }
        ]
        
        # 准备输入
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # 推理
        start_time = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        
        inference_time = time.time() - start_time
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"\n推理耗时: {inference_time:.2f}s")
        print(f"\n模型回答:\n{output_text}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  视频测试跳过: {e}")
        return False


def print_summary(results: dict):
    """打印测试摘要"""
    print("\n" + "=" * 60)
    print("  测试摘要")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test}: {status}")
    
    print(f"\n总计: {sum(1 for v in results.values() if v)}/{total} 通过")
    
    if all(results.values()):
        print("\n🎉 Qwen3-VL-2B 验证成功！模型可以正常使用。")
    else:
        print("\n⚠️  部分测试未通过，请检查上述错误信息。")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("   Qwen3-VL-2B-Instruct 验证脚本")
    print("=" * 60)
    
    # 1. 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装必要依赖")
        return
    
    # 2. 检查 GPU
    check_gpu()
    
    # 3. 下载模型
    model_path = download_model()
    if model_path is None:
        print("\n❌ 模型下载失败")
        return
    
    # 4. 加载模型
    try:
        model, processor = load_model(model_path)
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return
    
    # 5. 运行测试
    results = {}
    
    try:
        results["单图理解"] = test_single_image(model, processor)
    except Exception as e:
        print(f"❌ 单图测试失败: {e}")
        results["单图理解"] = False
    
    try:
        results["多图理解"] = test_multi_image(model, processor)
    except Exception as e:
        print(f"❌ 多图测试失败: {e}")
        results["多图理解"] = False
    
    # 视频测试可选
    try:
        results["视频理解"] = test_video(model, processor)
    except Exception as e:
        print(f"⚠️  视频测试跳过: {e}")
        results["视频理解"] = False
    
    # 6. 打印摘要
    print_summary(results)
    
    # 7. 清理
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n显存已释放")


if __name__ == "__main__":
    main()
