import numpy as np
import soundfile as sf
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
OUTPUT_DIR = ROOT_DIR / "static" / "voices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 生成一个简单的包含正弦波的参考音频
sr = 16000
duration = 5.0
t = np.linspace(0, duration, int(sr * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

# 生成多个版本以防万一
filenames = ["中文女.wav", "英文女.wav", "英文 女.wav"]
for name in filenames:
    output_path = OUTPUT_DIR / name
    sf.write(output_path, audio, sr)
    print(f"✅ Generated dummy reference audio at {output_path}")

print("\n🎉 Success. Please restart the TTS server.")
