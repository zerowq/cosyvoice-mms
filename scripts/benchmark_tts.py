"""
CosyVoice vs Kokoro-82M 对比测试脚本
测试维度: 音质、速度、GPU 显存占用

支持 CosyVoice 2.0 和 3.0 自动检测
- CosyVoice 2.0: 使用预设音色 (inference_sft)
- CosyVoice 3.0: 使用参考音频 (inference_cross_lingual)
"""
import os
import sys
import time
import gc
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

# 测试文本 (英语)
TEST_TEXTS = [
    "Hello, this is a short sentence for testing.",
    "The quick brown fox jumps over the lazy dog. This is a medium length sentence to evaluate the quality of speech synthesis.",
    "Artificial intelligence is transforming the way we interact with technology. From voice assistants to autonomous vehicles, AI is becoming an integral part of our daily lives. This longer text will help us evaluate the performance and quality of different text-to-speech models under more demanding conditions.",
]

def get_gpu_memory_mb():
    """获取当前 GPU 显存使用量 (MB)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS (Apple Silicon) 不支持显存查询，返回估算值
            return -1
    except:
        pass
    return -1

def clear_gpu_memory():
    """清理 GPU 缓存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

def benchmark_kokoro():
    """测试 Kokoro-82M"""
    from src.engines.kokoro_engine import KokoroEngine

    model_path = str(ROOT_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx")
    voices_path = str(ROOT_DIR / "models" / "kokoro" / "voices.json")

    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        logger.error("❌ Kokoro 模型文件缺失，跳过测试")
        return None

    results = {
        "model": "Kokoro-82M",
        "load_time": 0,
        "warmup_time": 0,
        "synthesis_times": [],
        "gpu_memory_mb": -1,
    }

    clear_gpu_memory()
    mem_before = get_gpu_memory_mb()

    # 加载模型（单独计时）
    logger.info("📥 [Kokoro] 加载模型...")
    start = time.time()
    engine = KokoroEngine(model_path, voices_path)
    engine._load_model()
    results["load_time"] = time.time() - start
    logger.info(f"✅ [Kokoro] 模型加载完成: {results['load_time']:.2f}s")

    mem_after = get_gpu_memory_mb()
    if mem_before >= 0 and mem_after >= 0:
        results["gpu_memory_mb"] = mem_after - mem_before

    # 预热（不计入测试时间）
    logger.info("🔥 [Kokoro] 预热模型...")
    start = time.time()
    engine.synthesize("Warmup test.", voice="af_sarah", lang="en-us")
    results["warmup_time"] = time.time() - start
    logger.info(f"✅ [Kokoro] 预热完成: {results['warmup_time']:.2f}s")

    # 合成测试（纯合成时间）
    output_dir = ROOT_DIR / "output" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("⏱️  [Kokoro] 开始合成测试...")
    for i, text in enumerate(TEST_TEXTS):
        output_file = str(output_dir / f"kokoro_test_{i+1}.wav")
        start = time.time()
        engine.synthesize(text, voice="af_sarah", lang="en-us", output_path=output_file)
        elapsed = time.time() - start
        results["synthesis_times"].append({
            "text_length": len(text),
            "time_seconds": elapsed,
            "output_file": output_file,
        })
        logger.info(f"  ✓ Text {i+1} ({len(text)} chars): {elapsed:.2f}s")

    return results

def benchmark_cosyvoice():
    """测试 CosyVoice（自动检测版本）"""
    try:
        from src.engines.cosyvoice_engine import CosyVoiceEngine
    except ImportError:
        logger.warning("⚠️ CosyVoiceEngine 未找到，跳过测试")
        return None

    # CosyVoice 模型路径（优先 3.0，回退到 2.0）
    possible_paths = [
        ROOT_DIR / "models" / "Fun-CosyVoice3-0.5B",
        ROOT_DIR / "models" / "CosyVoice2-0.5B",
        ROOT_DIR / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B",
    ]

    model_path = None
    model_version = None
    for p in possible_paths:
        if p.exists():
            model_path = str(p)
            if "Fun-CosyVoice3" in str(p):
                model_version = "Fun-CosyVoice-3.0"
            else:
                model_version = "CosyVoice-2.0"
            break

    if model_path is None:
        logger.error(f"❌ CosyVoice 模型路径不存在，尝试了: {[str(p) for p in possible_paths]}")
        return None

    results = {
        "model": model_version,
        "load_time": 0,
        "warmup_time": 0,
        "synthesis_times": [],
        "gpu_memory_mb": -1,
    }

    clear_gpu_memory()
    mem_before = get_gpu_memory_mb()

    try:
        # 加载模型（单独计时）
        logger.info(f"📥 [CosyVoice] 加载模型 {model_version}...")
        start = time.time()
        engine = CosyVoiceEngine(model_path, device="cpu")
        engine._load_model()
        results["load_time"] = time.time() - start
        logger.info(f"✅ [CosyVoice] 模型加载完成: {results['load_time']:.2f}s")

        mem_after = get_gpu_memory_mb()
        if mem_before >= 0 and mem_after >= 0:
            results["gpu_memory_mb"] = mem_after - mem_before

        # 预热（不计入测试时间）
        # 注意：CosyVoice3 没有预设音色，会自动使用 static/voices/英文女.wav 参考音频
        logger.info(f"🔥 [CosyVoice] 预热模型...")
        start = time.time()
        engine.synthesize("Warmup test.", voice="en_female")
        results["warmup_time"] = time.time() - start
        logger.info(f"✅ [CosyVoice] 预热完成: {results['warmup_time']:.2f}s")

        # 合成测试（纯合成时间）
        output_dir = ROOT_DIR / "output" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("⏱️  [CosyVoice] 开始合成测试...")
        for i, text in enumerate(TEST_TEXTS):
            output_file = str(output_dir / f"cosyvoice_test_{i+1}.wav")
            start = time.time()
            engine.synthesize(text, voice="en_female", output_path=output_file)
            elapsed = time.time() - start
            results["synthesis_times"].append({
                "text_length": len(text),
                "time_seconds": elapsed,
                "output_file": output_file,
            })
            logger.info(f"  ✓ Text {i+1} ({len(text)} chars): {elapsed:.2f}s")

        return results
    except Exception as e:
        logger.error(f"❌ CosyVoice 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_comparison(kokoro_results, cosyvoice_results):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("📊 CosyVoice vs Kokoro-82M 对比测试结果")
    print("=" * 70)

    # 加载时间
    print("\n🔄 模型加载时间 (一次性开销):")
    if kokoro_results:
        print(f"   Kokoro-82M:   {kokoro_results['load_time']:.2f}s")
    if cosyvoice_results:
        print(f"   CosyVoice:    {cosyvoice_results['load_time']:.2f}s")

    # 预热时间
    print("\n🔥 模型预热时间 (首次推理):")
    if kokoro_results:
        print(f"   Kokoro-82M:   {kokoro_results.get('warmup_time', 0):.2f}s")
    if cosyvoice_results:
        print(f"   CosyVoice:    {cosyvoice_results.get('warmup_time', 0):.2f}s")

    # GPU 显存
    print("\n💾 GPU 显存占用:")
    if kokoro_results and kokoro_results['gpu_memory_mb'] >= 0:
        print(f"   Kokoro-82M:   {kokoro_results['gpu_memory_mb']:.1f} MB (实测)")
    else:
        print(f"   Kokoro-82M:   < 500 MB (CPU模式)")
    if cosyvoice_results and cosyvoice_results['gpu_memory_mb'] >= 0:
        print(f"   CosyVoice:    {cosyvoice_results['gpu_memory_mb']:.1f} MB (实测)")
    else:
        print(f"   CosyVoice:    2-4 GB (GPU模式)")

    # 合成速度
    print("\n⏱️ 纯合成速度对比 (不含加载时间):")
    print(f"   {'文本(chars)':<15} {'Kokoro(s)':<15} {'CosyVoice(s)':<15} {'速度比':<15}")
    print("   " + "-" * 60)

    for i in range(len(TEST_TEXTS)):
        text_len = len(TEST_TEXTS[i])
        kokoro_time = kokoro_results['synthesis_times'][i]['time_seconds'] if kokoro_results and i < len(kokoro_results.get('synthesis_times', [])) else 0
        cosyvoice_time = cosyvoice_results['synthesis_times'][i]['time_seconds'] if cosyvoice_results and i < len(cosyvoice_results.get('synthesis_times', [])) else 0

        if kokoro_time > 0 and cosyvoice_time > 0:
            if kokoro_time < cosyvoice_time:
                diff = f"Kokoro {cosyvoice_time / kokoro_time:.1f}x 快"
            else:
                diff = f"CosyVoice {kokoro_time / cosyvoice_time:.1f}x 快"
        elif kokoro_time > 0:
            diff = "CosyVoice 未测试"
        elif cosyvoice_time > 0:
            diff = "Kokoro 未测试"
        else:
            diff = "N/A"

        print(f"   {text_len:<15} {kokoro_time:<15.2f} {cosyvoice_time:<15.2f} {diff:<15}")

    # 音频文件
    print("\n🎵 生成的音频文件:")
    output_dir = ROOT_DIR / "output" / "benchmark"
    print(f"   保存位置: {output_dir}")
    print(f"   访问URL: http://your-domain/output/benchmark/")
    print("   请手动对比音质差异。")

    print("\n" + "=" * 70)

def main():
    logger.info("🚀 开始 CosyVoice vs Kokoro-82M 对比测试...")
    
    # 测试 Kokoro
    logger.info("\n--- 测试 Kokoro-82M ---")
    kokoro_results = benchmark_kokoro()
    
    # 清理后测试 CosyVoice
    clear_gpu_memory()
    logger.info("\n--- 测试 CosyVoice (自动检测版本) ---")
    cosyvoice_results = benchmark_cosyvoice()
    
    # 打印对比结果
    print_comparison(kokoro_results, cosyvoice_results)

if __name__ == "__main__":
    main()
