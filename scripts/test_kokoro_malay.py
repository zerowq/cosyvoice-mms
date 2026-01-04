"""
Kokoro-82M 马来语发音测试
目的: 验证 Kokoro 用英语发音规则读马来语的效果
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.engines.kokoro_engine import KokoroEngine
from loguru import logger

# 马来语测试文本
MALAY_TEXTS = [
    "Selamat pagi.",  # 早上好
    "Terima kasih kerana menghubungi kami.",  # 感谢联系我们
    "Sila masukkan nombor akaun anda.",  # 请输入您的账号
    "Kami akan membantu anda secepat mungkin.",  # 我们会尽快帮助您
]

def test_kokoro_malay():
    model_path = str(ROOT_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx")
    voices_path = str(ROOT_DIR / "models" / "kokoro" / "voices.json")
    
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        logger.error("❌ Kokoro 模型文件缺失")
        return
    
    engine = KokoroEngine(model_path, voices_path)
    
    output_dir = ROOT_DIR / "output" / "malay_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧪 开始 Kokoro 马来语发音测试...")
    logger.info("📌 注意: Kokoro 不原生支持马来语，这里使用英语发音规则尝试")
    
    for i, text in enumerate(MALAY_TEXTS):
        output_file = str(output_dir / f"kokoro_malay_{i+1}.wav")
        logger.info(f"\n🎤 测试 {i+1}: {text}")
        
        try:
            # 尝试使用英语发音
            engine.synthesize(
                text=text, 
                voice="af_sarah",  # 使用英语音色
                lang="en-us",      # 使用英语发音规则
                output_path=output_file
            )
            logger.info(f"✅ 已保存: {output_file}")
        except Exception as e:
            logger.error(f"❌ 失败: {e}")
    
    logger.info(f"\n📂 所有音频已保存到: {output_dir}")
    logger.info("🎧 请手动播放这些音频，与 MMS-TTS 的马来语效果进行对比")
    logger.info("💡 预期: Kokoro 会用英语发音规则读马来语，可能听起来像'外国人说马来语'")

if __name__ == "__main__":
    test_kokoro_malay()
