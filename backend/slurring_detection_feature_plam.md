# Speech Slurring Detection — Feature Design & Implementation Plan
### Brain Stroke Early Detection System · Module 2 of 4

> **Scope:** This document covers only the *Speech Slurring / Dysarthria Detection* feature — one of four modules in the larger stroke-detection application. It defines all sub-features, the data strategy (Kaggle dysarthria dataset), the AI/ML technique rationale, and the full directory structure with per-file coding responsibilities.

---

## 1. Clinical Context

Speech slurring in a stroke context is clinically called **dysarthria** — a motor speech disorder caused by weakened or paralyzed muscles involved in articulation, phonation, and respiration. Stroke-induced dysarthria is characterised by:

- **Imprecise consonants** — blurred phoneme boundaries
- **Irregular articulatory breakdowns** — unexpected pauses or rate changes
- **Hypernasality / altered resonance**
- **Monopitch / reduced prosodic variation**
- **Short phrase length** and reduced breath support

The system must detect these acoustic signatures, quantify severity, produce a human-readable report, and output a calibrated **confidence score** and **risk score** — all in near-real-time.

---

## 2. Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — Dysarthria and Non-Dysarthria Speech Dataset](https://www.kaggle.com/datasets/poojag718/dysarthria-and-nondysarthria-speech-dataset) |
| Classes | Binary: Dysarthric / Non-Dysarthric |
| Format | `.wav` audio recordings |
| Supplementary | UA-Speech, TORGO (open-access research datasets for augmentation and cross-validation) |
| Augmentation | SpecAugment, time-stretching, pitch perturbation, additive noise (SNR 10–30 dB) |

---

## 3. Sub-Feature Breakdown

### SF-01 · Audio Ingestion & Preprocessing
Handles all input pathways (upload, mic stream, URL), validates audio quality, and produces a clean, normalized waveform ready for downstream analysis.

### SF-02 · Acoustic Feature Extraction
Extracts the multi-dimensional acoustic fingerprint of the speech signal — MFCCs, prosodic features (pitch, energy, rate), formant trajectories, and spectral descriptors — that serve as both model inputs and interpretable report signals.

### SF-03 · Deep Learning Inference Engine
The core model stack: a fine-tuned **HuBERT-large** transformer (self-supervised, pre-trained on healthy speech) with a Speaker-Agnostic Latent Regularisation (SALR) head that produces a raw dysarthria probability. A secondary **CNN–BiLSTM–Transformer** hybrid fuses spectrogram and acoustic features for robustness.

### SF-04 · Severity Grading & Slurring Score
Maps the model output to a graded severity scale (None / Mild / Moderate / Severe) and derives a continuous **Slurring Score (0–100)** using a calibrated sigmoid mapping over the ensemble confidence.

### SF-05 · Explainability & Segment Localisation
Uses **Grad-CAM** on the spectrogram branch and **attention rollout** on the transformer branch to highlight *which time segments* and *which acoustic dimensions* drove the prediction. Powers the visual report.

### SF-06 · Risk Score Computation
Combines the Slurring Score with contextual inputs (age, reported symptom onset, session baseline if available) via a rule-augmented logistic regression layer to produce a **Stroke Risk Score (0–100)** with a colour-coded tier (Low / Moderate / High / Critical).

### SF-07 · Report Generation Engine
Assembles a structured clinical-style PDF report: waveform plot, spectrogram with highlighted regions, per-feature breakdown table, score badges, plain-language interpretation, and a QR-linked disclaimer.

### SF-08 · Real-Time Streaming Mode *(optional v2)*
Enables continuous mic-based analysis using a chunked sliding-window pipeline so a caregiver can monitor speech in an ongoing conversation without uploading a file.

### SF-09 · API Layer & Service Contracts
FastAPI service exposing REST endpoints for all sub-features, with schema validation, auth middleware, and versioned contracts consumed by the frontend.

### SF-10 · Model Lifecycle & Monitoring
MLflow experiment tracking, model registry, data-drift detection (Evidently AI), and automated retraining triggers.

---

## 4. Technology Decisions (Rationale)

| Sub-feature | Technique | Why |
|---|---|---|
| SSL Feature Extraction | **HuBERT-large** (Facebook/Meta) | 2024–25 benchmarks show HuBERT outperforms wav2vec2-BASE and wav2vec2-LARGE on dysarthria detection by 1.33–2.86% accuracy and 6.54–10.46% on severity |
| Speaker-Invariance | **SALR** (triplet margin + cross-entropy multi-task) | Prevents the model from overfitting to speaker identity instead of pathology signals |
| Spectrogram Fusion | **CNN-BiLSTM-Transformer** with cross-attention | 2025 Nature paper reports 98.74–99.86% binary accuracy on TORGO/UA-Speech with wavelet scalograms + acoustic features |
| Explainability | **Grad-CAM** + attention rollout | Provides segment-level clinical evidence; Grad-CAM is used in validated clinical NLP/speech pipelines as of 2025 |
| Acoustic Baselines | **eGeMAPS** + **openSMILE** | Interpretable acoustic features that serve as fallback and explainability signal |
| Severity Calibration | **Platt scaling** on model logits | Ensures confidence scores are probabilistically calibrated, not raw logits |
| Streaming | **Silero VAD** + sliding window | Lightweight, production-proven VAD for chunked real-time inference |
| API | **FastAPI** + **Pydantic v2** | Async, typed, OpenAPI-native; standard for Python ML services in 2026 |
| Experiment Tracking | **MLflow** | Open-source, self-hostable, supports model registry and artifact tracking |
| Data Drift | **Evidently AI** | Detects distribution shift in incoming audio features vs. training baseline |

---

## 5. Directory Structure

```
speech_slurring_detection/
│
├── README.md
├── pyproject.toml                  # Poetry / uv project manifest
├── .env.example
├── .gitignore
│
├── configs/
│   ├── model_config.yaml           # HuBERT checkpoint, layer indices, SALR hyperparams
│   ├── feature_config.yaml         # MFCC bins, hop length, eGeMAPS params, formant settings
│   ├── inference_config.yaml       # Batch size, device (cuda/cpu/mps), precision (fp16/bf16)
│   ├── risk_score_config.yaml      # Severity thresholds, age-weight coefficients
│   └── logging_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── kaggle_dysarthria/      # Original Kaggle dataset (gitignored)
│   │   ├── ua_speech/              # UA-Speech supplement (gitignored)
│   │   └── torgo/                  # TORGO supplement (gitignored)
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── augmented/                  # SpecAugment + noise outputs
│   └── manifests/
│       ├── train_manifest.csv
│       ├── val_manifest.csv
│       └── test_manifest.csv
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/                  # SF-01
│   │   ├── __init__.py
│   │   ├── audio_loader.py         # Load wav/mp3/ogg/m4a; validate sample rate, duration
│   │   ├── preprocessor.py         # Resample to 16kHz mono, loudness normalisation (ITU-R BS.1770), silence trim
│   │   ├── vad.py                  # Silero VAD wrapper — strips leading/trailing silence
│   │   ├── quality_checker.py      # SNR estimation, clipping detection, minimum duration guard
│   │   └── schemas.py              # Pydantic models: AudioInput, PreprocessedAudio
│   │
│   ├── features/                   # SF-02
│   │   ├── __init__.py
│   │   ├── mfcc_extractor.py       # librosa MFCC + delta + delta-delta (39-dim)
│   │   ├── prosodic_extractor.py   # Fundamental frequency (CREPE model), energy, speaking rate, pause ratio
│   │   ├── formant_extractor.py    # F1/F2/F3 via Praat-parselmouth; vowel space area
│   │   ├── egemaps_extractor.py    # openSMILE eGeMAPS v02 feature set (88 dims)
│   │   ├── spectrogram_builder.py  # Log-mel spectrogram + wavelet CWT scalogram
│   │   ├── feature_fusion.py       # Concatenates all feature vectors; handles missing modalities
│   │   └── schemas.py              # FeatureBundle dataclass
│   │
│   ├── models/                     # SF-03
│   │   ├── __init__.py
│   │   ├── hubert_encoder.py       # HuBERT-large feature extractor; layer-weighted pooling
│   │   ├── salr_head.py            # SALR classification head: cross-entropy + triplet margin loss
│   │   ├── cnn_bilstm_transformer.py  # Hybrid spectrogram + acoustic feature encoder
│   │   ├── cross_attention_fusion.py  # Cross-attention between HuBERT and CNN branches
│   │   ├── ensemble.py             # Weighted ensemble of HuBERT-SALR and CNN-BiLSTM branches
│   │   ├── calibration.py          # Platt scaling; temperature scaling for confidence calibration
│   │   └── model_registry.py       # Load/save from MLflow registry; version pinning
│   │
│   ├── scoring/                    # SF-04
│   │   ├── __init__.py
│   │   ├── slurring_scorer.py      # Maps calibrated probability → Slurring Score 0–100
│   │   ├── severity_classifier.py  # Thresholded 4-class: None / Mild / Moderate / Severe
│   │   └── schemas.py              # SlurringResult dataclass
│   │
│   ├── explainability/             # SF-05
│   │   ├── __init__.py
│   │   ├── gradcam.py              # Grad-CAM on CNN spectrogram branch; returns time-freq heatmap
│   │   ├── attention_rollout.py    # Transformer attention rollout → frame-level importance
│   │   ├── segment_localiser.py    # Maps frame-level scores → labelled time segments (ms)
│   │   └── visualiser.py           # Matplotlib/Plotly: waveform, spectrogram overlay, heatmap PNG
│   │
│   ├── risk/                       # SF-06
│   │   ├── __init__.py
│   │   ├── risk_scorer.py          # Logistic regression over [slurring_score, age, onset_hours]
│   │   ├── risk_tier.py            # Enum: LOW / MODERATE / HIGH / CRITICAL + colour codes
│   │   └── schemas.py              # RiskAssessment dataclass
│   │
│   ├── reporting/                  # SF-07
│   │   ├── __init__.py
│   │   ├── report_builder.py       # Orchestrates all sections → ReportData object
│   │   ├── pdf_renderer.py         # ReportLab / WeasyPrint → styled PDF bytes
│   │   ├── json_renderer.py        # Machine-readable JSON report for API consumers
│   │   ├── templates/
│   │   │   ├── report_template.html   # Jinja2 HTML template for WeasyPrint
│   │   │   └── styles.css
│   │   └── schemas.py              # ReportOutput dataclass
│   │
│   ├── streaming/                  # SF-08 (v2)
│   │   ├── __init__.py
│   │   ├── stream_processor.py     # Chunked sliding-window (2s window, 0.5s hop)
│   │   ├── buffer_manager.py       # Ring buffer for overlapping chunk aggregation
│   │   └── websocket_handler.py   # FastAPI WebSocket endpoint handler
│   │
│   └── pipeline.py                 # Top-level orchestrator: chains SF-01 → SF-07
│
├── api/                            # SF-09
│   ├── __init__.py
│   ├── main.py                     # FastAPI app factory, lifespan, middleware registration
│   ├── routers/
│   │   ├── analyse.py              # POST /v1/speech/analyse — file upload → full report
│   │   ├── stream.py               # WS  /v1/speech/stream — real-time analysis
│   │   ├── health.py               # GET /healthz, GET /readyz
│   │   └── report.py               # GET /v1/report/{report_id} — fetch stored report
│   ├── middleware/
│   │   ├── auth.py                 # JWT / API-key validation
│   │   ├── rate_limiter.py         # Sliding-window rate limiting (Redis-backed)
│   │   └── request_id.py           # Inject X-Request-ID header
│   ├── dependencies.py             # FastAPI Depends(): model loader, DB session, storage client
│   └── schemas.py                  # Request/response Pydantic v2 models (OpenAPI-compatible)
│
├── training/                       # Offline training scripts
│   ├── __init__.py
│   ├── dataset.py                  # PyTorch Dataset: reads manifest CSVs, applies augmentation
│   ├── augmentation.py             # SpecAugment, time-stretch, pitch-shift, noise injection
│   ├── train_hubert_salr.py        # Fine-tune HuBERT with SALR loss; MLflow logging
│   ├── train_cnn_bilstm.py         # Train spectrogram hybrid model
│   ├── train_ensemble_weights.py   # Grid-search ensemble weight optimisation on val set
│   ├── calibrate.py                # Post-hoc Platt scaling on held-out set
│   └── evaluate.py                 # ACC, AUC, F1, sensitivity, specificity; confusion matrix
│
├── monitoring/                     # SF-10
│   ├── __init__.py
│   ├── drift_detector.py           # Evidently AI: compare live feature distribution vs. baseline
│   ├── metrics_logger.py           # Prometheus metrics exporter (inference latency, score distribution)
│   └── retraining_trigger.py       # Watches drift dashboard; creates JIRA/GitHub issue on threshold breach
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessor.py
│   │   ├── test_feature_extractors.py
│   │   ├── test_slurring_scorer.py
│   │   ├── test_risk_scorer.py
│   │   └── test_report_builder.py
│   ├── integration/
│   │   ├── test_pipeline_end_to_end.py
│   │   └── test_api_endpoints.py
│   └── fixtures/
│       ├── sample_dysarthric.wav
│       └── sample_healthy.wav
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training_experiments.ipynb
│   ├── 04_explainability_analysis.ipynb
│   └── 05_calibration_analysis.ipynb
│
├── docker/
│   ├── Dockerfile                  # Multi-stage: builder + slim runtime
│   ├── docker-compose.yml          # API + Redis + MLflow + Prometheus + Grafana
│   └── docker-compose.dev.yml
│
├── scripts/
│   ├── download_dataset.py         # Kaggle API download + integrity check
│   ├── build_manifests.py          # Stratified train/val/test split → CSV manifests
│   ├── run_training.sh
│   └── run_evaluation.sh
│
└── mlflow/
    └── mlruns/                     # Auto-generated by MLflow (gitignored except .gitkeep)
```

---

## 6. Data Flow (End-to-End)

```
[Audio Input: upload / mic / URL]
        │
        ▼
  SF-01 · Ingestion & Preprocessing
  (resample → normalise → VAD → quality gate)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
  SF-02 · Acoustic Features              SF-02 · Log-Mel Spectrogram
  (MFCC, eGeMAPS, prosody, formants)     + Wavelet CWT Scalogram
        │                                      │
        ▼                                      ▼
  HuBERT-large encoder              CNN-BiLSTM-Transformer encoder
  + SALR classification head        + cross-attention fusion
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              SF-03 · Ensemble Layer
              (weighted average of branch logits)
                       │
                       ▼
              SF-04 · Calibration (Platt Scaling)
              → Slurring Score [0–100]
              → Severity: None/Mild/Moderate/Severe
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
    SF-05 · Explainability   SF-06 · Risk Scorer
    (Grad-CAM + attention    (Slurring score + age +
     rollout → segments)      onset → Risk Score + Tier)
             │                    │
             └─────────┬──────────┘
                       ▼
              SF-07 · Report Generator
              (PDF + JSON: waveform, heatmap,
               scores, plain-language summary)
                       │
                       ▼
              SF-09 · API Response
              { report_url, slurring_score,
                severity, risk_score, risk_tier,
                confidence, segments[] }
```

---

## 7. Model Architecture Detail

### Primary Branch — HuBERT-SALR
```
Input waveform (16kHz, mono)
  └─► HuBERT-large feature encoder (24 transformer layers)
        └─► Weighted layer pooling (learnable α per layer)
              └─► SALR Head:
                    ├─► Linear(1024 → 256) → ReLU → Dropout(0.3)
                    ├─► Classification output: Linear(256 → 2)  [cross-entropy]
                    └─► Embedding projection: Linear(256 → 128) [triplet margin]
```

### Secondary Branch — CNN-BiLSTM-Transformer
```
Log-mel spectrogram + wavelet scalogram (stacked channels)
  └─► CNN feature extractor (3 × Conv2d blocks, BN, ReLU, MaxPool)
        └─► Flatten → feature sequence
              └─► BiLSTM (2 layers, hidden=256, bidirectional)
                    └─► Transformer encoder (4 heads, 2 layers)
                          └─► Cross-attention ← eGeMAPS feature vector
                                └─► Global average pool → Linear → Sigmoid
```

### Ensemble
```
p_final = α · p_hubert_salr + (1 - α) · p_cnn_bilstm
α optimised on validation set (expected ~0.6 based on literature)
```

---

## 8. Slurring Score & Risk Score Definitions

### Slurring Score (0–100)
```
raw_prob  = ensemble calibrated probability of dysarthria
slurring_score = round(raw_prob * 100, 1)

Severity tiers:
  0–20   → None      (green)
  21–45  → Mild      (yellow)
  46–70  → Moderate  (orange)
  71–100 → Severe    (red)
```

### Risk Score (0–100)
```
risk_score = σ(w0 + w1·slurring_score + w2·age_factor + w3·onset_factor) × 100

age_factor   = min(1.0, max(0.0, (age - 40) / 40))   # scales 0→1 for age 40–80
onset_factor = 1.0 if onset_hours ≤ 4.5 else 0.5     # golden window flag

Risk tiers:
  0–25   → Low
  26–50  → Moderate
  51–75  → High
  76–100 → Critical (triggers emergency alert flag in API response)
```

---

## 9. API Contract (Key Endpoint)

### `POST /v1/speech/analyse`

**Request** (multipart/form-data)
```
audio_file    : .wav / .mp3 / .ogg (≤ 60s, required)
patient_age   : int (optional, improves risk scoring)
onset_hours   : float (hours since symptom onset, optional)
language      : str (default: "en", ISO 639-1)
return_pdf    : bool (default: true)
```

**Response** (application/json)
```json
{
  "request_id": "uuid4",
  "slurring_score": 63.4,
  "severity": "moderate",
  "risk_score": 71.2,
  "risk_tier": "high",
  "confidence": 0.87,
  "segments": [
    { "start_ms": 420, "end_ms": 1150, "label": "imprecise_consonants", "weight": 0.82 },
    { "start_ms": 2300, "end_ms": 3100, "label": "irregular_rate", "weight": 0.71 }
  ],
  "acoustic_summary": {
    "speaking_rate_syllables_per_sec": 2.8,
    "pitch_variability_hz": 18.4,
    "pause_ratio": 0.34,
    "vowel_space_area": 0.61
  },
  "report_url": "https://api.example.com/v1/report/uuid4",
  "processing_time_ms": 1240,
  "model_version": "hubert-salr-v1.3"
}
```

---

## 10. Implementation Phases

### Phase 1 — Foundation (Weeks 1–3)
- Set up monorepo, Poetry env, Docker Compose stack
- Download and EDA of Kaggle dysarthria dataset (`notebooks/01`, `02`)
- Implement SF-01 (ingestion + VAD + quality check) with unit tests
- Implement SF-02 (all feature extractors) with unit tests
- Build train/val/test manifests with stratified split

### Phase 2 — Model Development (Weeks 4–7)
- Fine-tune HuBERT-large with SALR loss (`training/train_hubert_salr.py`)
- Train CNN-BiLSTM-Transformer branch (`training/train_cnn_bilstm.py`)
- Optimise ensemble weights on val set
- Apply Platt scaling calibration
- Log all experiments to MLflow (`notebooks/03`)

### Phase 3 — Scoring, Explainability & Reporting (Weeks 8–10)
- Implement SF-04 (scoring + severity) with calibrated thresholds
- Implement SF-05 (Grad-CAM + attention rollout + segment localiser)
- Implement SF-06 (risk scorer with configurable weights)
- Implement SF-07 (PDF + JSON report renderer)
- Integration test: full pipeline end-to-end (`tests/integration/`)

### Phase 4 — API & Hardening (Weeks 11–12)
- Build FastAPI service (SF-09) with auth, rate limiting, health checks
- Load test with Locust (target: p95 latency < 2s for 30s audio)
- Set up Evidently drift monitoring (SF-10)
- Write OpenAPI docs; generate client SDK

### Phase 5 — Streaming & v2 (Week 13+)
- Implement SF-08 (WebSocket streaming with sliding-window VAD)
- A/B test ensemble vs. single-branch in production shadow mode

---

## 11. Performance Targets

| Metric | Target |
|---|---|
| Binary detection AUC | ≥ 0.92 |
| Severity classification accuracy | ≥ 90% (4-class) |
| Inference latency (30s audio, GPU) | < 1.5s |
| Inference latency (30s audio, CPU) | < 5s |
| API p95 latency (end-to-end with report) | < 3s |
| False Negative Rate (dysarthric missed) | < 5% |

---

## 12. Key Dependencies

> **Runtime:** Python 3.11.x · All versions below are verified available on PyPI as of March 2026 and fully compatible with Python 3.11.

```toml
[tool.poetry.dependencies]
python = "^3.11"

# ── Core ML ────────────────────────────────────────────────────────────────
torch          = ">=2.6,<3.0"       # Stable on py311; CUDA 12.x wheels available
transformers   = ">=5.0,<6.0"       # HuBERT-large; v5 dropped py3.8 support
torchaudio     = ">=2.6,<3.0"       # Must match torch major.minor exactly

# ── Audio Processing ────────────────────────────────────────────────────────
librosa             = ">=0.11,<1.0"     # py311 wheels available from 0.10+
praat-parselmouth   = ">=0.4.7,<1.0"   # Formant extraction via Praat bindings
torchcrepe          = ">=0.0.24"        # Pitch (F0) estimation — PyTorch-native
                                        # Replaces `crepe` which requires TensorFlow
                                        # (TF + PyTorch dual-framework conflict on py311)
opensmile           = ">=2.6,<3.0"     # eGeMAPS v02 feature set (88 dims)
silero-vad          = ">=6.1,<7.0"     # Voice activity detection; v6 API is stable

# ── API ─────────────────────────────────────────────────────────────────────
fastapi    = ">=0.115,<1.0"    # Async REST + WebSocket; OpenAPI 3.1 native
pydantic   = ">=2.8,<3.0"     # v2 required (v1 is EOL); py311 fully supported
uvicorn    = { version = ">=0.30,<1.0", extras = ["standard"] }

# ── Reporting ───────────────────────────────────────────────────────────────
reportlab  = ">=4.2,<5.0"     # PDF generation (no system deps)
jinja2     = ">=3.1,<4.0"     # HTML templating for report renderer
matplotlib = ">=3.9,<4.0"     # Waveform / spectrogram plots
plotly     = ">=6.0,<7.0"     # Interactive heatmaps (server-side PNG export via kaleido)

# ── MLOps ───────────────────────────────────────────────────────────────────
mlflow     = ">=3.0,<4.0"     # v3 dropped py3.8; model registry + artifact tracking
evidently  = ">=0.7,<1.0"     # Data & feature drift detection

# ── Dev / Testing ────────────────────────────────────────────────────────────
[tool.poetry.group.dev.dependencies]
pytest      = ">=9.0,<10.0"
httpx       = ">=0.27,<1.0"   # Async FastAPI test client
pytest-asyncio = ">=0.24"
ruff        = ">=0.6"         # Linter + formatter (replaces flake8/black/isort)
mypy        = ">=1.11"
```

### Dependency Notes

| Package | Decision |
|---|---|
| `torchcrepe` instead of `crepe` | `crepe` depends on TensorFlow ≥2.x; running TF + PyTorch in the same environment on Python 3.11 causes protobuf and numpy ABI conflicts. `torchcrepe` is the official PyTorch-native port with an identical API surface. |
| `silero-vad >=6.1` | v5 had a breaking API change in `get_speech_timestamps`; v6 stabilised the interface and added Python 3.11 wheels. |
| `mlflow >=3.0` | MLflow 3.x dropped Python 3.8 support and added native `transformers` model flavour with HuBERT serialisation support. |
| `plotly >=6.0` | v6 rewrote the rendering engine; use with `kaleido>=0.2.1` for server-side static PNG/SVG export (no browser required). |
| `torch/torchaudio version lock` | These **must** share the same `major.minor` (e.g. both `2.6.x`). A mismatch causes silent runtime errors in `torchaudio` codec backends. |

---

*Document version: 1.1 · Python 3.11 verified · Prepared for engineering handoff · AI technique citations reflect published 2024–2025 research benchmarks.*