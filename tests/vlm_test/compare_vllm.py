"""
vLLM 推理性能对比测试：Qwen3.5-2B vs Qwen3-VL-2B
测试项目：
1. 模型加载时间和显存占用
2. 纯文本推理性能
3. 批处理性能
4. 首Token延迟 (TTFT)
"""

import os
import sys
import time
import gc
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 模型配置
MODELS = {
    "Qwen3-VL-2B": {
        "path": os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen3-VL-2B-Instruct"),
        "is_vl": True,  # 视觉语言模型
    },
    "Qwen3.5-2B": {
        "path": os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen3.5-2B"),
        "is_vl": False,  # 纯语言模型
    },
}


def clear_gpu():
    """清理 GPU 显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory():
    """获取当前 GPU 显存使用量"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / (1024**3)
    return 0


def test_model_vllm(model_name: str, model_config: dict):
    """使用 vLLM 测试单个模型"""
    print("\n" + "=" * 60)
    print(f"测试模型: {model_name}")
    print("=" * 60)
    
    from vllm import LLM, SamplingParams
    
    model_path = model_config["path"]
    
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return None
    
    # 加载模型
    print("加载模型...")
    start_load = time.time()
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="float16",  # Tesla T4 不支持 bfloat16
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        disable_log_stats=True,
    )
    
    load_time = time.time() - start_load
    memory_gb = get_gpu_memory()
    
    print(f"加载时间: {load_time:.2f}s, 显存: {memory_gb:.2f} GB")
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "is_vl": model_config["is_vl"],
        "load_time": load_time,
        "memory_gb": memory_gb,
    }
    
    # ============ 测试1: 纯文本推理 ============
    print("\n[测试1] 纯文本推理性能")
    print("-" * 50)
    
    text_prompts = [
        "请介绍一下人工智能的发展历史。",
        "什么是深度学习？请简要说明。",
        "解释一下神经网络的工作原理。",
    ]
    
    sampling_params = SamplingParams(max_tokens=128, temperature=0.7)
    
    text_results = []
    for i, prompt in enumerate(text_prompts):
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        inference_time = time.time() - start_time
        
        output_tokens = len(outputs[0].outputs[0].token_ids)
        tps = output_tokens / inference_time
        
        print(f"  测试 {i+1}: {inference_time:.2f}s, {output_tokens} tokens, {tps:.1f} tokens/s")
        text_results.append({
            "time": inference_time,
            "tokens": output_tokens,
            "tps": tps,
            "output": outputs[0].outputs[0].text[:100],
        })
    
    results["text_avg_time"] = sum(r["time"] for r in text_results) / len(text_results)
    results["text_avg_tps"] = sum(r["tps"] for r in text_results) / len(text_results)
    results["text_results"] = text_results
    
    # ============ 测试2: 首Token延迟 (TTFT) ============
    print("\n[测试2] 首Token延迟测试")
    print("-" * 50)
    
    ttft_prompts = [
        "请用一句话回答：什么是机器学习？",
        "简短回答：Python是什么？",
        "一句话解释：什么是API？",
    ]
    
    ttft_results = []
    # 使用流式输出来测量 TTFT
    from vllm import SamplingParams
    
    for i, prompt in enumerate(ttft_prompts):
        # vLLM 不直接支持 TTFT，使用短生成来估计
        sampling_params_ttft = SamplingParams(max_tokens=1, temperature=0.0)
        
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params_ttft)
        ttft = time.time() - start_time
        
        print(f"  测试 {i+1}: TTFT = {ttft:.3f}s")
        ttft_results.append(ttft)
    
    results["avg_ttft"] = sum(ttft_results) / len(ttft_results)
    
    # ============ 测试3: 批处理性能 ============
    print("\n[测试3] 批处理性能")
    print("-" * 50)
    
    batch_prompts = [f"请简短回答：什么是机器学习？变体{i+1}" for i in range(10)]
    sampling_params_batch = SamplingParams(max_tokens=64, temperature=0.7)
    
    # 顺序处理
    print("顺序处理 10 个请求...")
    start_time = time.time()
    total_tokens_seq = 0
    for p in batch_prompts:
        outputs = llm.generate([p], sampling_params_batch)
        total_tokens_seq += len(outputs[0].outputs[0].token_ids)
    seq_time = time.time() - start_time
    seq_throughput = total_tokens_seq / seq_time
    
    # 批处理
    print("批处理 10 个请求...")
    start_time = time.time()
    outputs = llm.generate(batch_prompts, sampling_params_batch)
    batch_time = time.time() - start_time
    total_tokens_batch = sum(len(o.outputs[0].token_ids) for o in outputs)
    batch_throughput = total_tokens_batch / batch_time
    
    speedup = seq_time / batch_time
    
    print(f"  顺序处理: {seq_time:.2f}s, {seq_throughput:.1f} tokens/s")
    print(f"  批处理:   {batch_time:.2f}s, {batch_throughput:.1f} tokens/s")
    print(f"  加速比:   {speedup:.2f}x")
    
    results["batch_seq_time"] = seq_time
    results["batch_time"] = batch_time
    results["batch_speedup"] = speedup
    results["batch_throughput"] = batch_throughput
    
    # ============ 测试4: 不同长度输出性能 ============
    print("\n[测试4] 不同输出长度性能")
    print("-" * 50)
    
    length_tests = [
        ("短输出 (32 tokens)", 32),
        ("中输出 (128 tokens)", 128),
        ("长输出 (256 tokens)", 256),
    ]
    
    length_results = []
    for name, max_tokens in length_tests:
        prompt = "请详细介绍一下人工智能。"
        sampling_params_len = SamplingParams(max_tokens=max_tokens, temperature=0.7)
        
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params_len)
        inference_time = time.time() - start_time
        
        output_tokens = len(outputs[0].outputs[0].token_ids)
        tps = output_tokens / inference_time
        
        print(f"  {name}: {inference_time:.2f}s, {output_tokens} tokens, {tps:.1f} tokens/s")
        length_results.append({
            "name": name,
            "max_tokens": max_tokens,
            "actual_tokens": output_tokens,
            "time": inference_time,
            "tps": tps,
        })
    
    results["length_results"] = length_results
    
    # 清理
    del llm
    clear_gpu()
    
    return results


def print_comparison(results: dict):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("  性能对比汇总：Qwen3.5-2B vs Qwen3-VL-2B")
    print("=" * 70)
    
    model_names = list(results.keys())
    
    # 基本信息对比
    print("\n### 模型基本信息")
    print("-" * 55)
    print(f"{'指标':<20} {model_names[0]:>17} {model_names[1]:>17}")
    print("-" * 55)
    print(f"{'模型类型':<20} {'VL模型':>17} {'纯语言模型':>17}")
    print(f"{'加载时间 (s)':<20} {results[model_names[0]]['load_time']:>17.2f} {results[model_names[1]]['load_time']:>17.2f}")
    print(f"{'显存占用 (GB)':<20} {results[model_names[0]]['memory_gb']:>17.2f} {results[model_names[1]]['memory_gb']:>17.2f}")
    
    # 纯文本推理对比
    print("\n### 纯文本推理性能")
    print("-" * 55)
    print(f"{'指标':<20} {model_names[0]:>17} {model_names[1]:>17}")
    print("-" * 55)
    print(f"{'平均延迟 (s)':<20} {results[model_names[0]]['text_avg_time']:>17.2f} {results[model_names[1]]['text_avg_time']:>17.2f}")
    print(f"{'吞吐量 (tokens/s)':<20} {results[model_names[0]]['text_avg_tps']:>17.1f} {results[model_names[1]]['text_avg_tps']:>17.1f}")
    print(f"{'平均TTFT (s)':<20} {results[model_names[0]]['avg_ttft']:>17.3f} {results[model_names[1]]['avg_ttft']:>17.3f}")
    
    # 批处理性能对比
    print("\n### 批处理性能 (10个请求)")
    print("-" * 55)
    print(f"{'指标':<20} {model_names[0]:>17} {model_names[1]:>17}")
    print("-" * 55)
    print(f"{'批处理时间 (s)':<20} {results[model_names[0]]['batch_time']:>17.2f} {results[model_names[1]]['batch_time']:>17.2f}")
    print(f"{'批处理吞吐 (tok/s)':<20} {results[model_names[0]]['batch_throughput']:>17.1f} {results[model_names[1]]['batch_throughput']:>17.1f}")
    print(f"{'批处理加速比':<20} {results[model_names[0]]['batch_speedup']:>17.2f}x {results[model_names[1]]['batch_speedup']:>17.2f}x")
    
    # 不同长度输出对比
    print("\n### 不同输出长度性能 (tokens/s)")
    print("-" * 55)
    print(f"{'输出长度':<20} {model_names[0]:>17} {model_names[1]:>17}")
    print("-" * 55)
    for i, name in enumerate(["短输出 (32)", "中输出 (128)", "长输出 (256)"]):
        tps0 = results[model_names[0]]['length_results'][i]['tps']
        tps1 = results[model_names[1]]['length_results'][i]['tps']
        print(f"{name:<20} {tps0:>17.1f} {tps1:>17.1f}")
    
    # 性能差异分析
    print("\n### 性能差异分析")
    print("-" * 55)
    
    speed_diff = (results[model_names[1]]['text_avg_tps'] - results[model_names[0]]['text_avg_tps']) / results[model_names[0]]['text_avg_tps'] * 100
    ttft_diff = (results[model_names[1]]['avg_ttft'] - results[model_names[0]]['avg_ttft']) / results[model_names[0]]['avg_ttft'] * 100
    
    print(f"吞吐量差异: {model_names[1]} 比 {model_names[0]} {'快' if speed_diff > 0 else '慢'} {abs(speed_diff):.1f}%")
    print(f"TTFT差异: {model_names[1]} 比 {model_names[0]} {'快' if ttft_diff < 0 else '慢'} {abs(ttft_diff):.1f}%")
    
    # 输出示例
    print("\n### 输出示例对比")
    print("-" * 55)
    for model_name in model_names:
        print(f"\n[{model_name}]:")
        output = results[model_name]['text_results'][0]['output']
        print(f"  {output}...")


def main():
    print("\n" + "=" * 70)
    print("   vLLM 推理性能对比：Qwen3.5-2B vs Qwen3-VL-2B")
    print("=" * 70)
    
    # GPU 信息
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"CUDA Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    
    # 检查模型
    available_models = {}
    for model_name, config in MODELS.items():
        if os.path.exists(config["path"]):
            available_models[model_name] = config
            print(f"✓ 找到模型: {model_name}")
        else:
            print(f"✗ 模型不存在: {config['path']}")
    
    if len(available_models) < 2:
        print("\n错误：需要两个模型都存在才能进行对比")
        return
    
    results = {}
    
    # 测试每个模型
    for model_name, config in available_models.items():
        result = test_model_vllm(model_name, config)
        if result:
            results[model_name] = result
        clear_gpu()
    
    # 打印对比结果
    if len(results) == 2:
        print_comparison(results)
    
    print("\n" + "=" * 70)
    print("  测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
