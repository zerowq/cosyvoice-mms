"""
TTS系统配置
"""
import os
import torch
from dataclasses import dataclass
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

# 强制设置 ModelScope 缓存目录到项目内的 models 目录
os.environ["MODELSCOPE_CACHE"] = str(ROOT_DIR / "models")
# 强制开启离线模式，禁止任何网络请求 (CI 环境/离线生产必选)
os.environ["MODELSCOPE_OFFLINE"] = "1"
os.environ["MODELSCOPE_ENVIRONMENT"] = "local"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 【终极离线补丁】拦截 ModelScope 的所有可能网络请求
def patch_environment():
    try:
        # 1. 拦截 snapshot_download
        import modelscope.hub.snapshot_download as ms_download
        original_snapshot = ms_download.snapshot_download
        
        def mocked_snapshot(model_id, *args, **kwargs):
            if "wetext" in model_id or "cosyvoice" in model_id.lower():
                local_path = ROOT_DIR / "models" / "hub" / model_id.split('/')[-1]
                if not local_path.exists():
                     local_path = ROOT_DIR / "models" / "hub" / model_id.replace('/', os.sep)
                if "wetext" in model_id:
                    local_wetext = ROOT_DIR / "models" / "hub" / "pengzhendong" / "wetext"
                    if local_wetext.exists(): return str(local_wetext)
                if local_path.exists(): return str(local_path)
            return original_snapshot(model_id, *args, **kwargs)
        ms_download.snapshot_download = mocked_snapshot

        # 2. 彻底拦截 HubApi (版本检查的源头)
        from modelscope.hub.api import HubApi
        HubApi.login = lambda *args, **kwargs: None
        HubApi.get_model_revisions = lambda *args, **kwargs: ["master"]

        # 检查关键模型文件是否存在，如果不存在则在离线模式下提示错误
        models_dir = ROOT_DIR / "models"
        if not (models_dir / "CosyVoice2-0.5B").exists():
            print(f"❌ CRITICAL ERROR: Model CosyVoice2-0.5B not found in {models_dir}")
            print("Please ensure models are pre-downloaded via scripts/download_models.py during build.")
    except Exception:
        pass

patch_environment()

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class TTSConfig:
    """TTS配置类"""
    
    # 模型路径
    cosyvoice_model_path: str = str(ROOT_DIR / "models" / "CosyVoice2-0.5B")
    mms_model_dir: str = str(ROOT_DIR / "models")
    
    # 设备配置
    device: str = get_default_device()
    
    # 路径配置
    cache_dir: str = str(ROOT_DIR / "tts_cache")
    output_dir: str = str(ROOT_DIR / "output")
    static_dir: str = str(ROOT_DIR / "static")
    log_dir: str = str(ROOT_DIR / "logs")
    
    # 默认声音 (CosyVoice 预设 ID 通常为中文命名，支持跨语言)
    default_english_voice: str = "中文女"
    default_malay_voice: str = "default"
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    def __post_init__(self):
        # 确保所有必要目录存在
        for path in [self.cache_dir, self.output_dir, self.static_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(self.static_dir, "audio"), exist_ok=True)

# 全局配置实例
config = TTSConfig()
