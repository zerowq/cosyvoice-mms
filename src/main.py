"""
TTS Service FastAPI 主程序
"""
import uvicorn
import uuid
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional

from src.config import config
from src.core.service import get_service

# 启动时检查并下载模型
def ensure_models_ready():
    """启动时检查模型，如果缺失则自动下载"""
    try:
        from scripts.download_models import check_models_exist, main as download_main

        all_exist, missing = check_models_exist()

        if all_exist:
            print("✅ All models are ready!")
            return True

        print(f"⚠️  Missing models detected: {', '.join(missing)}")
        print("📥 Starting automatic model download...")

        success = download_main(auto_download=True)

        if success:
            print("✅ Models downloaded successfully!")
            return True
        else:
            print("❌ Failed to download models")
            print("Please run: python scripts/download_models.py")
            return False

    except Exception as e:
        print(f"⚠️  Warning: Could not check models: {e}")
        print("Please ensure models are in the 'models/' directory")
        return False

app = FastAPI(
    title="TTS Service API",
    description="英文 (CosyVoice 2.0) + 马来文 (MMS) TTS服务",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory=config.static_dir), name="static")
app.mount("/output", StaticFiles(directory=config.output_dir), name="output")

class TTSRequest(BaseModel):
    text: str
    language: Literal["en", "ms", "kokoro", "kokoro_ms"] = "en"
    voice: Optional[str] = None

class TTSResponse(BaseModel):
    success: bool
    engine: str
    audio_url: str
    cached: bool

@app.get("/")
async def root():
    return {"service": "TTS Service", "status": "running"}

@app.get("/api/health")
async def health():
    service = get_service()
    return service.get_health()

@app.post("/api/tts", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    try:
        filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(config.static_dir, "audio", filename)
        
        service = get_service()
        result = service.synthesize(
            text=request.text,
            language=request.language,
            voice=request.voice,
            output_path=output_path
        )
        
        return TTSResponse(
            success=True,
            engine=result["engine"],
            audio_url=f"/static/audio/{filename}",
            cached=result["cached"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts/stream")
async def synthesize_stream(request: TTSRequest):
    """
    流式合成语音 (适用于 Chatbot 实时播报)
    返回 raw pcm 字节流
    """
    try:
        service = get_service()
        gen = service.stream_synthesize(
            text=request.text,
            language=request.language,
            voice=request.voice
        )
        return StreamingResponse(gen, media_type="audio/pcm")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def metrics():
    """获取服务性能指标 (目前仅包含延迟)"""
    service = get_service()
    return service.get_metrics()

if __name__ == "__main__":
    import uvicorn

    # 启动前检查模型
    print("=" * 60)
    print("TTS Service Startup")
    print("=" * 60)
    ensure_models_ready()
    print("=" * 60)

    # 修改默认端口为 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)
