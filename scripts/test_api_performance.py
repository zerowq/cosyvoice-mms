#!/usr/bin/env python3
"""
API 性能测试脚本
测试已启动服务的真实响应速度（不含模型加载时间）
"""
import requests
import time
import sys
from pathlib import Path

# 测试文本
TEST_TEXTS = [
    ("短句", "Hello, this is a short sentence for testing."),
    ("中句", "The quick brown fox jumps over the lazy dog. This is a medium length sentence to evaluate the quality of speech synthesis."),
    ("长句", "Artificial intelligence is transforming the way we interact with technology. From voice assistants to autonomous vehicles, AI is becoming an integral part of our daily lives. This longer text will help us evaluate the performance and quality of different text-to-speech models under more demanding conditions."),
]

def test_tts_api(base_url: str, language: str = "en", warmup: bool = True):
    """
    测试TTS API性能

    Args:
        base_url: API基础URL (例如: http://localhost:8080)
        language: 语言代码 (en/ms/kokoro)
        warmup: 是否进行预热请求
    """
    api_url = f"{base_url}/api/tts"

    print("=" * 70)
    print(f"📊 TTS API 性能测试")
    print(f"🌐 服务地址: {base_url}")
    print(f"🗣️  语言: {language}")
    print("=" * 70)

    # 检查服务健康状态
    try:
        health_url = f"{base_url}/api/health"
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        print(f"\n✅ 服务健康检查通过: {response.json()}")
    except Exception as e:
        print(f"\n❌ 服务健康检查失败: {e}")
        print(f"请确认服务已启动并可访问 {base_url}")
        return

    # 预热请求
    if warmup:
        print("\n🔥 执行预热请求...")
        try:
            warmup_start = time.time()
            response = requests.post(
                api_url,
                json={
                    "text": "Warmup request.",
                    "language": language
                },
                timeout=30
            )
            warmup_time = time.time() - warmup_start
            if response.status_code == 200:
                print(f"✅ 预热完成: {warmup_time:.2f}s")
            else:
                print(f"⚠️  预热请求失败: {response.status_code}")
        except Exception as e:
            print(f"⚠️  预热请求出错: {e}")

    # 性能测试
    print("\n⏱️  开始性能测试...")
    print(f"{'文本类型':<10} {'字符数':<10} {'响应时间(s)':<15} {'音频URL':<50}")
    print("-" * 70)

    results = []
    for name, text in TEST_TEXTS:
        try:
            start_time = time.time()
            response = requests.post(
                api_url,
                json={
                    "text": text,
                    "language": language
                },
                timeout=60
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                audio_url = data.get("audio_url", "N/A")
                print(f"{name:<10} {len(text):<10} {elapsed:<15.2f} {audio_url:<50}")
                results.append({
                    "name": name,
                    "text_length": len(text),
                    "time": elapsed,
                    "success": True
                })
            else:
                print(f"{name:<10} {len(text):<10} {'ERROR':<15} Status: {response.status_code}")
                results.append({
                    "name": name,
                    "text_length": len(text),
                    "time": 0,
                    "success": False
                })
        except Exception as e:
            print(f"{name:<10} {len(text):<10} {'ERROR':<15} {str(e)[:35]}")
            results.append({
                "name": name,
                "text_length": len(text),
                "time": 0,
                "success": False
            })

    # 统计结果
    print("\n" + "=" * 70)
    print("📈 测试统计:")
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_time = sum(r["time"] for r in successful_results) / len(successful_results)
        min_time = min(r["time"] for r in successful_results)
        max_time = max(r["time"] for r in successful_results)
        print(f"   成功请求: {len(successful_results)}/{len(results)}")
        print(f"   平均响应时间: {avg_time:.2f}s")
        print(f"   最快响应: {min_time:.2f}s")
        print(f"   最慢响应: {max_time:.2f}s")
    else:
        print("   ❌ 所有请求均失败")
    print("=" * 70)

def test_streaming_api(base_url: str, language: str = "en"):
    """测试流式API性能"""
    api_url = f"{base_url}/api/tts/stream"

    print("\n" + "=" * 70)
    print(f"📊 流式API 性能测试")
    print("=" * 70)

    test_text = "Hello! This is a streaming test."

    try:
        print(f"\n⏱️  测试文本: {test_text}")
        start_time = time.time()

        response = requests.post(
            api_url,
            json={
                "text": test_text,
                "language": language
            },
            stream=True,
            timeout=30
        )

        # 接收首字节时间
        ttfb = None
        total_bytes = 0

        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                if ttfb is None:
                    ttfb = time.time() - start_time
                total_bytes += len(chunk)

        total_time = time.time() - start_time

        print(f"✅ 首字节时间(TTFB): {ttfb:.2f}s")
        print(f"✅ 总响应时间: {total_time:.2f}s")
        print(f"✅ 接收数据: {total_bytes / 1024:.2f} KB")

    except Exception as e:
        print(f"❌ 流式请求失败: {e}")

    print("=" * 70)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="TTS API 性能测试")
    parser.add_argument("--url", default="http://localhost:8080", help="API基础URL")
    parser.add_argument("--language", default="en", choices=["en", "ms", "kokoro", "kokoro_ms"], help="语言代码")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热请求")
    parser.add_argument("--stream", action="store_true", help="同时测试流式API")

    args = parser.parse_args()

    # 测试普通API
    test_tts_api(args.url, args.language, warmup=not args.no_warmup)

    # 测试流式API
    if args.stream:
        test_streaming_api(args.url, args.language)

if __name__ == "__main__":
    main()
