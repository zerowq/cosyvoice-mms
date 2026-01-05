# CosyVoice-MMS

CosyVoice 2.0 + MMS-TTS - High-quality Text-to-Speech service supporting English and Malay languages.

## Features

- English TTS using CosyVoice 2.0 with zero-shot voice cloning
- Malay TTS using Meta MMS
- GPU-accelerated inference
- Streaming audio generation
- RESTful API
- Offline/air-gapped deployment support

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

```bash
# Start the service
python src/main.py
```

The service will be available at `http://localhost:8080`

## API Documentation

- Health check: `GET /api/health`
- Metrics: `GET /api/metrics`
- Test page: `http://localhost:8080/static/test_stream.html`

## Requirements

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (recommended)
- CUDA support

## License

See LICENSE file for details.
