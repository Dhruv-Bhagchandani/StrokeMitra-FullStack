# Speech Slurring Detection System

> **Brain Stroke Early Detection · Module 2 of 4**
> Clinical-grade dysarthria detection using deep learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

This system detects **speech slurring (dysarthria)** from audio recordings using a dual-branch deep learning architecture:

- **HuBERT-SALR**: Self-supervised speech transformer with speaker-agnostic latent regularization
- **CNN-BiLSTM-Transformer**: Spectrogram and acoustic feature fusion

**Key Features:**
- 📊 **Slurring Score (0-100)** with severity grading (None/Mild/Moderate/Severe)
- 🎯 **Risk Score** incorporating age and symptom onset timing
- 🔍 **Explainability** via Grad-CAM and attention rollout
- 📄 **Clinical Reports** in PDF/JSON format
- 🚀 **REST API** with auth, rate limiting, and health checks
- 🔬 **MLOps** with MLflow tracking and drift detection

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Poetry** (recommended) or pip
- **Redis** (for rate limiting)
- **FFmpeg** (for audio processing)
- **CUDA 12.x** (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
cd speech_slurring_detection

# Install dependencies with Poetry
poetry install

# Or with pip
pip install -e .

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### Configuration

Update [.env](.env) with your settings:
- `DEVICE`: cuda / mps / cpu (auto-detected if empty)
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `REDIS_URL`: Redis connection string
- `API_SECRET_KEY`: Secret key for API authentication

---

## Usage

### 1. API Server

Start the FastAPI service:

```bash
# Development mode with hot reload
poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API documentation available at:
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 2. Analyze Audio File

```bash
curl -X POST http://localhost:8000/v1/speech/analyse \
  -H "X-API-Key: your-api-key" \
  -F "audio_file=@path/to/audio.wav" \
  -F "patient_age=65" \
  -F "onset_hours=2.5" \
  -F "return_pdf=true"
```

**Response:**
```json
{
  "request_id": "uuid",
  "slurring_score": 63.4,
  "severity": "moderate",
  "risk_score": 71.2,
  "risk_tier": "high",
  "confidence": 0.87,
  "segments": [
    {
      "start_ms": 420,
      "end_ms": 1150,
      "label": "imprecise_consonants",
      "weight": 0.82
    }
  ],
  "acoustic_summary": {
    "speaking_rate_syllables_per_sec": 2.8,
    "pitch_variability_hz": 18.4,
    "pause_ratio": 0.34,
    "vowel_space_area": 0.61
  },
  "report_url": "http://localhost:8000/v1/report/uuid",
  "processing_time_ms": 1240,
  "model_version": "ensemble-v1.0"
}
```

### 3. Python API

```python
from src.pipeline import SlurringDetectionPipeline

# Initialize pipeline
pipeline = SlurringDetectionPipeline(config_dir="configs/")

# Analyze audio
result = pipeline.analyse(
    audio_file="path/to/audio.wav",
    patient_age=65,
    onset_hours=2.5
)

print(f"Slurring Score: {result.slurring_score}")
print(f"Severity: {result.severity}")
print(f"Risk Score: {result.risk_score}")
```

---

## Training

### 1. Download Dataset

```bash
# Set Kaggle credentials in .env
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

# Download Kaggle dysarthria dataset
poetry run python scripts/download_dataset.py

# Build train/val/test manifests
poetry run python scripts/build_manifests.py
```

### 2. Train Models

```bash
# Fine-tune HuBERT with SALR loss
poetry run python training/train_hubert_salr.py

# Train CNN-BiLSTM-Transformer branch
poetry run python training/train_cnn_bilstm.py

# Optimize ensemble weights
poetry run python training/train_ensemble_weights.py

# Calibrate confidence scores
poetry run python training/calibrate.py

# Or run full pipeline
bash scripts/run_training.sh
```

### 3. Evaluate

```bash
poetry run python training/evaluate.py --model ensemble --split test
```

**Metrics:**
- Binary detection AUC ≥ 0.92
- Severity classification accuracy ≥ 90%
- False negative rate < 5%

---

## Project Structure

```
speech_slurring_detection/
├── configs/                  # YAML configuration files
│   ├── feature_config.yaml
│   ├── model_config.yaml
│   ├── inference_config.yaml
│   └── risk_score_config.yaml
├── src/
│   ├── ingestion/           # SF-01: Audio loading, preprocessing, VAD
│   ├── features/            # SF-02: MFCC, prosody, formants, eGeMAPS
│   ├── models/              # SF-03: HuBERT-SALR, CNN-BiLSTM, ensemble
│   ├── scoring/             # SF-04: Slurring score, severity classification
│   ├── explainability/      # SF-05: Grad-CAM, attention rollout
│   ├── risk/                # SF-06: Risk score computation
│   ├── reporting/           # SF-07: PDF/JSON report generation
│   ├── streaming/           # SF-08: Real-time streaming (v2)
│   └── pipeline.py          # End-to-end orchestration
├── api/
│   ├── routers/             # FastAPI endpoints
│   ├── middleware/          # Auth, rate limiting, request ID
│   └── main.py              # FastAPI app factory
├── training/                # Training scripts
├── monitoring/              # Drift detection, metrics
├── tests/                   # Unit and integration tests
├── notebooks/               # Jupyter notebooks for exploration
└── docker/                  # Docker setup
```

---

## Docker Deployment

### Development

```bash
docker compose -f docker/docker-compose.dev.yml up
```

### Production

```bash
# Build image
docker compose -f docker/docker-compose.yml build

# Start services
docker compose -f docker/docker-compose.yml up -d

# Services available at:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

---

## Testing

```bash
# Run all tests
poetry run pytest

# Unit tests only
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# With coverage
poetry run pytest --cov=src --cov-report=html
```

---

## Architecture

### Data Flow

```
Audio Input
    ↓
[SF-01] Ingestion & Preprocessing
    ↓
[SF-02] Feature Extraction
    ├─→ HuBERT-SALR Branch
    └─→ CNN-BiLSTM Branch
    ↓
[SF-03] Ensemble Model
    ↓
[SF-04] Calibration & Scoring
    ↓
[SF-05] Explainability
    ↓
[SF-06] Risk Assessment
    ↓
[SF-07] Report Generation
    ↓
API Response (JSON/PDF)
```

### Model Architecture

**HuBERT-SALR Branch:**
- HuBERT-large (24 transformer layers)
- Layer-weighted pooling
- SALR head: classification + embedding

**CNN-BiLSTM Branch:**
- CNN on log-mel + wavelet spectrogram
- BiLSTM (2 layers, 256 hidden)
- Transformer encoder (4 heads, 2 layers)
- Cross-attention with eGeMAPS features

**Ensemble:** Weighted average (α=0.6 for HuBERT, 0.4 for CNN)

---

## Configuration

All parameters are configurable via YAML files in [configs/](configs/):

- **[feature_config.yaml](configs/feature_config.yaml)**: MFCC bins, mel bins, F0 range
- **[model_config.yaml](configs/model_config.yaml)**: HuBERT checkpoint, SALR dims, ensemble alpha
- **[inference_config.yaml](configs/inference_config.yaml)**: Device, precision, batch size
- **[risk_score_config.yaml](configs/risk_score_config.yaml)**: Severity thresholds, risk tiers

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/speech/analyse` | Analyze audio file |
| GET | `/v1/report/{report_id}` | Retrieve generated report |
| WS | `/v1/speech/stream` | Real-time streaming analysis (v2) |
| GET | `/healthz` | Liveness check |
| GET | `/readyz` | Readiness check |
| GET | `/docs` | Interactive API documentation |

---

## Monitoring

### MLflow

Track experiments and model versions:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

View at http://localhost:5000

### Prometheus + Grafana

Monitor API metrics:

```bash
docker compose up prometheus grafana
```

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Data Drift Detection

```python
from monitoring.drift_detector import DriftDetector

detector = DriftDetector()
report = detector.detect_drift(reference_data, current_data)
```

---

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Binary detection AUC | ≥ 0.92 | 🎯 Target |
| Severity accuracy (4-class) | ≥ 90% | 🎯 Target |
| Inference latency (30s audio, GPU) | < 1.5s | 🎯 Target |
| API p95 latency (end-to-end) | < 3s | 🎯 Target |
| False negative rate | < 5% | 🎯 Target |

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{speech_slurring_detection_2026,
  title={Speech Slurring Detection System for Stroke Early Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/speech-slurring-detection}
}
```

---

## License

[Specify your license]

---

## Clinical Disclaimer

⚠️ **This system is for research and development purposes only.**

- Not FDA-approved or clinically validated
- Not intended for diagnostic use
- Should not replace professional medical evaluation
- Always consult a healthcare provider for stroke symptoms

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## Support

- **Documentation**: [See plan](slurring_detection_feature_plam.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/speech-slurring-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/speech-slurring-detection/discussions)

---

**Built with:** Python 3.11 · PyTorch 2.6 · HuBERT-large · FastAPI · MLflow
