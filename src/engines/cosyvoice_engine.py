"""
CosyVoice TTS 引擎封装（支持 v2.0 和 v3.0）
与官方 CosyVoice 652132e 版本保持一致
"""
import os
import sys
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Generator

# 获取根目录
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
COSYVOICE_PATH = ROOT_DIR / "CosyVoice"
MODELS_DIR = ROOT_DIR / "models"

# 强制设置 ModelScope 缓存目录，确保能找到 download_models.py 下载的 wetext 资源
os.environ["MODELSCOPE_CACHE"] = str(MODELS_DIR)
# 开启 ModelScope 离线模式，禁止联网检查更新
os.environ["MODELSCOPE_OFFLINE"] = "true"

# 重要：必须在导入 CosyVoice 之前添加 Matcha-TTS 路径
# 这样才能正确导入 matcha.models.components.flow_matching 等模块
if str(COSYVOICE_PATH / "third_party" / "Matcha-TTS") not in sys.path:
    sys.path.insert(0, str(COSYVOICE_PATH / "third_party" / "Matcha-TTS"))

# 然后添加 CosyVoice 路径
if str(COSYVOICE_PATH) not in sys.path:
    sys.path.insert(0, str(COSYVOICE_PATH))


class CosyVoiceEngine:
    """CosyVoice TTS引擎（自动检测 v2.0 或 v3.0）"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._loaded = False
        # 根据模型路径判断版本
        self._is_v3 = "CosyVoice3" in model_path or "Fun-CosyVoice" in model_path

    def _load_model(self):
        if not self._loaded:
            try:
                import torch
                # 为了保证生成音频的正确性（避免破音/乱码），暂时强制关闭 fp16
                # use_fp16 = torch.cuda.is_available()
                use_fp16 = False

                if self._is_v3:
                    from cosyvoice.cli.cosyvoice import CosyVoice3
                    print(f"🔄 Loading CosyVoice 3.0 on {self.device} (fp16={use_fp16})...")
                    self._model = CosyVoice3(
                        self.model_path,
                        load_trt=False,
                        fp16=use_fp16
                    )
                    print(f"✅ CosyVoice 3.0 loaded on {self.device} (fp16={use_fp16})!")
                else:
                    from cosyvoice.cli.cosyvoice import CosyVoice2
                    print(f"🔄 Loading CosyVoice 2.0 on {self.device} (fp16={use_fp16})...")
                    self._model = CosyVoice2(
                        self.model_path,
                        load_jit=True,
                        load_trt=False,
                        fp16=use_fp16
                    )
                    print(f"✅ CosyVoice 2.0 loaded on {self.device} (fp16={use_fp16})!")

                self._loaded = True
            except Exception as e:
                print(f"❌ Failed to load CosyVoice: {e}")
                raise
        return self._model

    @property
    def model(self):
        return self._load_model()

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    def list_voices(self) -> list:
        return self.model.list_available_spks()

    def synthesize(self, text: str, voice: str = "英文女", output_path: Optional[str] = None, stream: bool = False) -> torch.Tensor:
        """
        合成语音
        - 如果 voice 是预设音色，使用 inference_sft
        - 否则使用参考音频进行 inference_cross_lingual
        """
        audio_chunks = []

        try:
            model = self.model
            spk_list = model.list_available_spks()

            if voice in spk_list:
                # 使用预设音色
                iterable = model.inference_sft(text, voice, stream=stream)
            else:
                # 使用参考音频进行跨语言合成
                ref_audio_path = self._find_reference_audio(voice)
                if ref_audio_path:
                    print(f"🎤 Using reference audio: {ref_audio_path}")
                    # 直接传递路径，官方代码内部会处理加载和重采样
                    iterable = model.inference_cross_lingual(text, ref_audio_path, stream=stream)
                else:
                    if spk_list:
                        print(f"⚠️ Voice '{voice}' not found, fallback to '{spk_list[0]}'")
                        iterable = model.inference_sft(text, spk_list[0], stream=stream)
                    else:
                        raise ValueError(f"Voice '{voice}' not found and no reference audio at static/voices/{voice}.wav")

            for result in iterable:
                audio_chunks.append(result['tts_speech'])

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"CosyVoice inference failed: {e}")

        audio = torch.cat(audio_chunks, dim=1) if len(audio_chunks) > 1 else audio_chunks[0]

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torchaudio.save(output_path, audio, self.sample_rate)
        return audio

    def synthesize_stream(self, text: str, voice: str = "英文女") -> Generator[bytes, None, None]:
        """
        流式合成音频
        """
        try:
            model = self.model
            spk_list = model.list_available_spks()

            if voice in spk_list:
                iterable = model.inference_sft(text, voice, stream=True)
            else:
                ref_audio_path = self._find_reference_audio(voice)
                if ref_audio_path:
                    print(f"🎤 [CosyVoice] Using reference audio: {os.path.basename(ref_audio_path)}")
                    iterable = model.inference_cross_lingual(text, ref_audio_path, stream=True)
                else:
                    print(f"⚠️ [CosyVoice] Voice '{voice}' not found, falling back to first available")
                    if spk_list:
                        iterable = model.inference_sft(text, spk_list[0], stream=True)
                    else:
                        raise ValueError(f"No voices available")

            for chunk in iterable:
                speech = chunk['tts_speech'].numpy().flatten()
                yield speech.tobytes()

        except Exception as e:
            print(f"❌ [CosyVoice] Streaming error: {e}")
            raise

    def _find_reference_audio(self, voice: str) -> Optional[str]:
        """
        查找参考音频文件
        """
        voice_dir = os.path.join(ROOT_DIR, "static", "voices")

        # 尝试多种文件名格式
        candidates = [
            f"{voice.strip()}.wav",
            f"{voice.strip().replace(' ', '')}.wav",
            f"{voice.strip().lower()}.wav",
        ]

        for filename in candidates:
            path = os.path.join(voice_dir, filename)
            if os.path.exists(path):
                return path

        return None
