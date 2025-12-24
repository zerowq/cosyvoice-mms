#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import json
import os

BASE_URL = "http://localhost:8000"
OUTPUT_DIR = "output_test"

def test_health():
    print("--- 1. Testing Health Check ---")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}, Response: {response.json()}")

def test_tts(text, language, voice=None):
    print(f"\n--- Testing {language} TTS: '{text[:20]}...' ---")
    payload = {
        "text": text,
        "language": language,
        "voice": voice
    }
    response = requests.post(f"{BASE_URL}/api/tts", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Engine: {data['engine']}, Cached: {data['cached']}")
        print(f"Audio URL: {data['audio_url']}")
        return data['audio_url']
    else:
        print(f"❌ Failed: {response.status_code}, Detail: {response.text}")
        return None

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 检查健康状态
    test_health()
    
    print("\n⚠️ Note: Synthesis will fail if models are not downloaded yet.")
    
    # 2. 测试英文 (CosyVoice)
    test_tts("Hello, this is a test from my Mac M1 using CosyVoice 2.0.", "en", "英文女")
    
    # 3. 测试马来文 (MMS)
    test_tts("Selamat pagi, apa khabar hari ini?", "ms")
