# CosyVoice-MMS

CosyVoice 2.0 + MMS-TTS - High-quality Text-to-Speech service supporting English and Malay languages.

## Features

- English TTS using CosyVoice 2.0 with zero-shot voice cloning
- Malay TTS using Meta MMS
- GPU-accelerated inference
- Streaming audio generation
- RESTful API
- Offline/air-gapped deployment support
- **Automatic model download on first startup**

## Quick Start

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Installation

```bash
# Clone the repository
git clone https://github.com/zerowq/cosyvoice-mms.git
cd cosyvoice-mms

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Automatic Model Download (Recommended)

Models will be automatically downloaded on first startup:

```bash
# Start the service (models will auto-download if missing)
python src/main.py
```

The service will:
1. Check if models exist in `models/` directory
2. Automatically download missing models from ModelScope/HuggingFace
3. Start the TTS service on `http://localhost:8080`

### Option 2: Manual Model Download

If you prefer to download models separately:

```bash
# Download all models
python scripts/download_models.py

# Then start the service
python src/main.py
```

## API Documentation

- Health check: `GET /api/health`
- Metrics: `GET /api/metrics`
- Test page: `http://localhost:8080/static/test_stream.html`

## Requirements

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (recommended)
- CUDA support
- Internet connection (for first-time model download)

## Model Download

Models are automatically downloaded from:
- **CosyVoice 2.0**: ModelScope (iic/CosyVoice2-0.5B)
- **MMS-TTS**: HuggingFace (facebook/mms-tts-eng, facebook/mms-tts-zlm)

Total model size: ~8-10GB

## License

See LICENSE file for details.
