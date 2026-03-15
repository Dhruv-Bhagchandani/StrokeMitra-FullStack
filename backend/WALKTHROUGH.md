# 🎓 Complete System Walkthrough

> **A comprehensive guide to understanding your Speech Slurring Detection system**

This document walks you through the entire system architecture, explains how each component works, and shows you the data flow from audio input to final report.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Data Flow: Audio to Report](#data-flow-audio-to-report)
3. [Module-by-Module Deep Dive](#module-by-module-deep-dive)
4. [How to Use the System](#how-to-use-the-system)
5. [Understanding the Code](#understanding-the-code)
6. [Extending the System](#extending-the-system)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│  Audio File (.wav, .mp3, .ogg, .m4a) + Optional Context        │
│  (patient_age, onset_hours)                                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY POINTS                                 │
│  ┌──────────────────┐         ┌─────────────────────┐          │
│  │  Python API      │         │    REST API         │          │
│  │  (Direct Call)   │         │  (FastAPI Server)   │          │
│  └─────────┬────────┘         └──────────┬──────────┘          │
└────────────┼───────────────────────────────┼───────────────────┘
             └───────────────┬───────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│               SLURRING DETECTION PIPELINE                       │
│                   (src/pipeline.py)                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ SF-01: Audio Ingestion & Preprocessing                  │  │
│  │ • Load audio (AudioLoader)                              │  │
│  │ • Preprocess (AudioPreprocessor)                        │  │
│  │ • Voice Activity Detection (VoiceActivityDetector)      │  │
│  │ • Quality Check (QualityChecker)                        │  │
│  └────────────────────────┬────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ SF-02: Feature Extraction                               │  │
│  │ • MFCC (39-dim)          MFCCExtractor                  │  │
│  │ • Prosody (F0, energy)   ProsodicExtractor             │  │
│  │ • Formants (F1/F2/F3)    FormantExtractor              │  │
│  │ • eGeMAPS (88-dim)       EGeMAPSExtractor              │  │
│  │ • Spectrogram            SpectrogramBuilder            │  │
│  │ • Fusion                 FeatureFusion                 │  │
│  └────────────────────────┬────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ SF-03: Model Inference                                  │  │
│  │ • Ensemble Model (HuBERT-SALR + CNN-BiLSTM)            │  │
│  │ • Calibration (Platt Scaling)                          │  │
│  │ • Model Registry (loads models)                        │  │
│  │ [MVP: Uses placeholder/mock predictions]               │  │
│  └────────────────────────┬────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ SF-04: Slurring Scoring                                 │  │
│  │ • Compute Slurring Score (0-100)                       │  │
│  │ • Classify Severity (None/Mild/Moderate/Severe)        │  │
│  └────────────────────────┬────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ SF-05: Explainability (Simplified in MVP)               │  │
│  │ • Segment Localization (mock segments)                 │  │
│  │ • Time-based annotations                               │  │
│  └────────────────────────┬────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ SF-06: Risk Assessment                                  │  │
│  │ • Compute Risk Score (0-100)                           │  │
│  │ • Apply age and onset factors                          │  │
│  │ • Classify Risk Tier (Low/Moderate/High/Critical)      │  │
│  └────────────────────────┬────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ SF-07: Report Generation                                │  │
│  │ • Build Report Data                                     │  │
│  │ • Render JSON Report                                    │  │
│  │ • [PDF rendering not implemented in MVP]               │  │
│  └────────────────────────┬────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                   │
│  • Slurring Score (0-100)                                      │
│  • Severity Level (None/Mild/Moderate/Severe)                 │
│  • Risk Score (0-100)                                          │
│  • Risk Tier (Low/Moderate/High/Critical)                     │
│  • Acoustic Summary (speaking rate, pitch, pauses, etc.)      │
│  • Annotated Segments                                          │
│  • JSON Report                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Audio Processing** | librosa, soundfile | Load and manipulate audio |
| **VAD** | Silero VAD v6 | Voice activity detection |
| **Feature Extraction** | librosa, torchcrepe, parselmouth, OpenSMILE | Acoustic features |
| **Deep Learning** | PyTorch, Transformers | Model architecture |
| **Scoring** | NumPy, SciPy | Score computation |
| **API** | FastAPI, Pydantic v2 | REST API |
| **Data Validation** | Pydantic v2 | Schema validation |

---

## Data Flow: Audio to Report

Let's trace a single audio file through the entire system:

### Step-by-Step Example

```python
# INPUT: audio_file = "patient_speech.wav", patient_age = 65, onset_hours = 2.5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: Audio Ingestion (SF-01)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1.1: Load audio
AudioLoader.load("patient_speech.wav")
→ AudioInput(
    file_name="patient_speech.wav",
    sample_rate=44100,
    duration_sec=15.3,
    num_channels=2
  )
→ waveform: np.array([0.01, -0.02, 0.03, ...])  # 673,380 samples

# 1.2: Quality check
QualityChecker.check(waveform)
→ QualityMetrics(
    snr_db=22.4,
    clipping_ratio=0.002,
    is_valid=True,
    quality_issues=[]
  )

# 1.3: Preprocess
AudioPreprocessor.process(waveform, sr=44100)
→ Resample 44100Hz → 16000Hz
→ Normalize loudness to -23 LUFS
→ Trim silence (30dB threshold)
→ PreprocessedAudio(
    waveform: np.array([...]),  # 244,800 samples
    sample_rate=16000,
    duration_sec=15.3,
    resampled=True,
    normalized=True,
    trimmed=True
  )

# 1.4: Voice Activity Detection
VoiceActivityDetector.detect_speech(waveform)
→ speech_segments = [(0.2, 4.8), (5.1, 9.7), (10.2, 14.9)]
→ speech_ratio = 0.89 (89% speech)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: Feature Extraction (SF-02)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 2.1: MFCC
MFCCExtractor.extract(waveform, sr=16000)
→ MFCCFeatures(
    mfcc: (13, 479),        # 13 coefficients × 479 time frames
    delta: (13, 479),
    delta_delta: (13, 479),
    combined: (39, 479),    # Total 39-dimensional features
    mean: (39,),
    std: (39,)
  )

# 2.2: Prosody
ProsodicExtractor.extract(waveform, sr=16000)
→ ProsodicFeatures(
    f0_mean=148.2,          # Hz (adult male typical: 85-180Hz)
    f0_std=24.3,
    voicing_ratio=0.71,     # 71% voiced
    energy_mean=0.042,
    speaking_rate_syllables_per_sec=2.8,
    pause_ratio=0.11,       # 11% pauses
    num_pauses=4
  )

# 2.3: Formants
FormantExtractor.extract(waveform, sr=16000)
→ FormantFeatures(
    f1_mean=720.0,          # Hz (F1: vowel height)
    f2_mean=1240.0,         # Hz (F2: vowel frontness)
    f3_mean=2630.0,         # Hz (F3: rounding)
    vowel_space_area=245000.0  # Hz² (acoustic space)
  )

# 2.4: eGeMAPS
EGeMAPSExtractor.extract(waveform, sr=16000)
→ EGeMAPSFeatures(
    features: (88,)         # 88-dimensional vector
    # Includes: pitch, jitter, shimmer, loudness, spectral features
  )

# 2.5: Spectrogram
SpectrogramBuilder.build(waveform, sr=16000)
→ SpectrogramFeatures(
    log_mel: (128, 479),    # 128 mel bands × 479 time frames
    wavelet_scalogram: (128, 479),
    stacked: (2, 128, 479)  # 2-channel input for CNN
  )

# 2.6: Feature Fusion
FeatureFusion.fuse(feature_bundle)
→ fused_acoustic: (145,)    # 39 (MFCC) + 10 (prosody) + 8 (formants) + 88 (eGeMAPS)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: Model Inference (SF-03) [MVP: Placeholder]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 3.1: Ensemble Model (Placeholder)
EnsembleModel.predict(
    waveform=(244800,),
    spectrogram=(2, 128, 479),
    acoustic_features=(145,)
)
→ {
    "logits": [-0.52, 0.41],           # [healthy, dysarthric]
    "probabilities": [0.42, 0.58],
    "raw_probability": 0.58            # 58% dysarthric
  }

# 3.2: Calibration
PlattScaling.transform(raw_probability=0.58)
→ calibrated_probability = 0.58  # Identity transform in MVP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: Slurring Scoring (SF-04)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 4.1: Compute Slurring Score
SlurringScorer.compute_score(
    calibrated_probability=0.58,
    confidence=0.58
)
→ SlurringResult(
    slurring_score=58.0,      # 0.58 × 100
    severity=SeverityLevel.MODERATE,  # 58 falls in 46-70 range
    confidence=0.58,
    raw_probability=0.58,
    calibrated_probability=0.58,
    model_version="ensemble-v1.0-placeholder"
  )

# Severity thresholds:
# 0-20:   None
# 21-45:  Mild
# 46-70:  Moderate  ← 58 falls here
# 71-100: Severe

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: Explainability (SF-05) [MVP: Simplified]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Mock segments for MVP
→ segments = [
    SegmentAnnotation(
      start_ms=0,
      end_ms=5100,
      label="imprecise_consonants",
      weight=0.82
    ),
    SegmentAnnotation(
      start_ms=5100,
      end_ms=10200,
      label="irregular_rate",
      weight=0.71
    ),
    SegmentAnnotation(
      start_ms=10200,
      end_ms=15300,
      label="monopitch",
      weight=0.65
    )
  ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: Risk Assessment (SF-06)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 6.1: Compute factors
age_factor = (65 - 40) / (80 - 40) = 0.625      # Age scaling
onset_factor = 1.0                               # Within 4.5h window

# 6.2: Logistic regression
logit = w0 + w1×slurring + w2×age_factor + w3×onset_factor
      = -2.5 + 0.05×58 + 1.2×0.625 + 0.8×1.0
      = -2.5 + 2.9 + 0.75 + 0.8
      = 1.95

risk_probability = sigmoid(1.95) = 0.875
risk_score = 0.875 × 100 = 87.5

# 6.3: Classify tier
RiskAssessment(
    risk_score=87.5,
    risk_tier=RiskTier.CRITICAL,  # 87.5 > 75 threshold
    emergency_alert=True,          # Critical tier triggers alert
    slurring_score=58.0,
    age_factor=0.625,
    onset_factor=1.0,
    patient_age=65,
    onset_hours=2.5
  )

# Risk tier thresholds:
# 0-25:   Low
# 26-50:  Moderate
# 51-75:  High
# 76-100: Critical  ← 87.5 falls here

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: Report Generation (SF-07)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 7.1: Build report data
ReportBuilder.build(
    slurring_result=...,
    risk_assessment=...,
    acoustic_summary={
      "speaking_rate_syllables_per_sec": 2.8,
      "pitch_mean_hz": 148.2,
      "pitch_variability_hz": 24.3,
      "pause_ratio": 0.11,
      "vowel_space_area": 0.61
    },
    segments=[...],
    processing_time_ms=1240
)

# 7.2: Render JSON
JSONRenderer.render(report_data)
→ {
    "report_id": "a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6",
    "slurring_score": 58.0,
    "severity": "moderate",
    "risk_score": 87.5,
    "risk_tier": "critical",
    "emergency_alert": true,
    "confidence": 0.58,
    "segments": [...],
    "acoustic_summary": {...},
    "processing_time_ms": 1240,
    "model_version": "ensemble-v1.0-placeholder"
  }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

return {
  "slurring_score": 58.0,       # MODERATE severity
  "risk_score": 87.5,            # CRITICAL risk
  "emergency_alert": True,        # ⚠ EMERGENCY
  "recommendation": "Call 911 immediately"
}
```

---

## Module-by-Module Deep Dive

### SF-01: Audio Ingestion

**Purpose:** Load, validate, and preprocess audio files.

**Key Files:**
- `src/ingestion/audio_loader.py`
- `src/ingestion/preprocessor.py`
- `src/ingestion/vad.py`
- `src/ingestion/quality_checker.py`

**What It Does:**

```python
# Example: Processing an audio file
from src.ingestion.audio_loader import AudioLoader
from src.ingestion.preprocessor import AudioPreprocessor

# 1. Load
loader = AudioLoader()
audio_input, waveform = loader.load("speech.wav")
# → waveform shape: (samples,), e.g., (673380,) for 15.3s @ 44.1kHz

# 2. Preprocess
preprocessor = AudioPreprocessor(target_sr=16000)
preprocessed = preprocessor.process(waveform, sr=44100, original_duration=15.3)
# → Resampled to 16kHz, normalized, silence trimmed
# → New shape: (244800,) for 15.3s @ 16kHz
```

**Key Concepts:**

1. **Resampling:** Converts all audio to 16kHz (standard for speech processing)
2. **Loudness Normalization:** ITU-R BS.1770 standard (-23 LUFS target)
3. **VAD:** Removes leading/trailing silence, detects speech segments
4. **Quality Checks:** Ensures SNR > 10dB, clipping < 1%, duration 5-60s

---

### SF-02: Feature Extraction

**Purpose:** Extract acoustic fingerprints from audio.

**Key Files:**
- `src/features/mfcc_extractor.py` - Mel-frequency cepstral coefficients
- `src/features/prosodic_extractor.py` - Pitch, energy, rhythm
- `src/features/formant_extractor.py` - Vowel resonances
- `src/features/egemaps_extractor.py` - 88-dim acoustic feature set
- `src/features/spectrogram_builder.py` - Visual representation
- `src/features/feature_fusion.py` - Combine everything

**What Each Feature Captures:**

| Feature | What It Measures | Dysarthria Indicators |
|---------|------------------|----------------------|
| **MFCC** (39-dim) | Spectral envelope shape | Imprecise articulation, blurred phonemes |
| **F0 (Pitch)** | Fundamental frequency | Monopitch, reduced prosody |
| **Energy** | Loudness contour | Reduced breath support |
| **Speaking Rate** | Syllables per second | Slowed or irregular rate |
| **Pauses** | Silence ratio | Unexpected pauses |
| **Formants** (F1/F2/F3) | Vowel resonances | Hypernasality, vowel space reduction |
| **eGeMAPS** (88-dim) | Comprehensive acoustic set | Jitter, shimmer, spectral features |
| **Spectrogram** | Time-frequency representation | Visual patterns for CNN |

**Example:**

```python
from src.features.prosodic_extractor import ProsodicExtractor

extractor = ProsodicExtractor()
prosody = extractor.extract(waveform, sr=16000)

print(f"Average pitch: {prosody.f0_mean:.1f} Hz")
print(f"Speaking rate: {prosody.speaking_rate_syllables_per_sec:.1f} syl/s")
print(f"Pause ratio: {prosody.pause_ratio:.1%}")

# Healthy speech typically:
# - F0 mean: 85-180 Hz (male), 165-255 Hz (female)
# - Speaking rate: 4-6 syllables/second
# - Pause ratio: 10-20%

# Dysarthric speech shows:
# - Reduced F0 variability (monotone)
# - Slower rate: 2-4 syllables/second
# - Increased pauses: 25-40%
```

---

### SF-03: Model Inference

**Purpose:** Predict dysarthria probability from features.

**Key Files:**
- `src/models/ensemble.py` - Dual-branch ensemble
- `src/models/model_registry.py` - Load models
- `src/models/calibration.py` - Probability calibration

**Architecture (When Trained):**

```
Branch 1: HuBERT-SALR
  Waveform → HuBERT-large (24 transformer layers)
           → Layer-weighted pooling
           → SALR head (classification + embedding)
           → Logits [healthy, dysarthric]

Branch 2: CNN-BiLSTM-Transformer
  Spectrogram → CNN (3 conv blocks)
              → BiLSTM (2 layers, hidden=256)
              → Transformer (4 heads, 2 layers)
              → Cross-attention with eGeMAPS
              → Logits [healthy, dysarthric]

Ensemble:
  Final = α × HuBERT_logits + (1-α) × CNN_logits
  where α = 0.6 (configurable)
```

**MVP Placeholder Behavior:**

```python
# In MVP, models return random but realistic-looking predictions
from src.models.ensemble import EnsembleModel

model = EnsembleModel(alpha=0.6)
prediction = model.predict(waveform, spectrogram, acoustic_features)

# Returns:
# {
#   "logits": [-0.52, 0.41],           # Mock logits
#   "probabilities": [0.42, 0.58],     # Softmax of logits
#   "raw_probability": 0.58            # P(dysarthric)
# }

# After training, these will be real predictions!
```

---

### SF-04: Slurring Scoring

**Purpose:** Convert model output to clinical scores.

**Key Files:**
- `src/scoring/slurring_scorer.py`
- `src/scoring/severity_classifier.py`

**Scoring Logic:**

```python
# Slurring Score = Calibrated Probability × 100
slurring_score = calibrated_prob * 100  # 0.58 → 58.0

# Severity Classification
if slurring_score <= 20:
    severity = "none"      # Green
elif slurring_score <= 45:
    severity = "mild"      # Yellow
elif slurring_score <= 70:
    severity = "moderate"  # Orange
else:
    severity = "severe"    # Red
```

**Clinical Interpretation:**

- **None (0-20):** Speech patterns within normal limits
- **Mild (21-45):** Slight articulatory imprecision, generally intelligible
- **Moderate (46-70):** Noticeable slurring, reduced intelligibility
- **Severe (71-100):** Marked dysarthria, significantly impaired

---

### SF-06: Risk Assessment

**Purpose:** Estimate stroke risk using speech + clinical context.

**Key Files:**
- `src/risk/risk_scorer.py`
- `src/risk/risk_tier.py`

**Risk Model:**

```python
# Logistic regression formula
risk_score = sigmoid(
    w0 +                          # Baseline: -2.5
    w1 × slurring_score +         # Speech contribution: 0.05
    w2 × age_factor +             # Age contribution: 1.2
    w3 × onset_factor             # Timing contribution: 0.8
) × 100

# Age factor: scales 0-1 for ages 40-80
age_factor = (age - 40) / (80 - 40)
age_factor = max(0, min(1, age_factor))

# Onset factor: urgency based on golden window
if onset_hours <= 4.5:
    onset_factor = 1.0  # Within golden window: URGENT
else:
    onset_factor = 0.5  # Outside window: less urgent
```

**Example Calculation:**

```python
# Patient: 65 years old, symptoms started 2.5 hours ago, slurring score = 58

age_factor = (65 - 40) / (80 - 40) = 0.625
onset_factor = 1.0  # Within 4.5h window

logit = -2.5 + 0.05×58 + 1.2×0.625 + 0.8×1.0
      = -2.5 + 2.9 + 0.75 + 0.8
      = 1.95

risk_score = sigmoid(1.95) × 100 = 87.5

# Risk Tier:
# 0-25:   Low       → Continue monitoring
# 26-50:  Moderate  → Evaluate within 24-48h
# 51-75:  High      → Seek evaluation within 12h
# 76-100: Critical  → EMERGENCY: Call 911 immediately
```

---

### SF-07: Reporting

**Purpose:** Generate structured reports for clinicians.

**Key Files:**
- `src/reporting/report_builder.py`
- `src/reporting/json_renderer.py`

**Report Structure:**

```json
{
  "report_id": "uuid",
  "generated_at": "2026-03-15T10:30:00Z",

  "summary": {
    "slurring_score": 58.0,
    "severity": "moderate",
    "risk_score": 87.5,
    "risk_tier": "critical",
    "emergency_alert": true
  },

  "slurring_analysis": {
    "score": 58.0,
    "severity": "moderate",
    "confidence": 0.87,
    "clinical_note": "Moderate dysarthric features present..."
  },

  "risk_assessment": {
    "score": 87.5,
    "tier": "critical",
    "emergency_alert": true,
    "action": "EMERGENCY: Call 911 immediately",
    "contributing_factors": {
      "slurring_score": 58.0,
      "age_factor": 0.625,
      "onset_factor": 1.0,
      "within_golden_window": true
    }
  },

  "acoustic_summary": {
    "speaking_rate_syllables_per_sec": 2.8,
    "pitch_mean_hz": 148.2,
    "pitch_variability_hz": 24.3,
    "pause_ratio": 0.11,
    "vowel_space_area": 0.61
  },

  "segments": [
    {
      "start_ms": 0,
      "end_ms": 5100,
      "label": "imprecise_consonants",
      "weight": 0.82,
      "time_range": "0.00s - 5.10s"
    }
  ]
}
```

---

## How to Use the System

### Method 1: Python API (Direct)

```python
from src.pipeline import SlurringDetectionPipeline

# Initialize (happens once)
pipeline = SlurringDetectionPipeline()

# Analyze audio
result = pipeline.analyse(
    audio_file="patient_recording.wav",
    patient_age=72,           # Optional
    onset_hours=3.2,          # Optional
    return_report=True        # Generate full JSON report
)

# Access results
print(f"Slurring Score: {result['slurring_score']}")
print(f"Severity: {result['severity']}")
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Tier: {result['risk_tier']}")
print(f"Emergency: {result['emergency_alert']}")

# Access acoustic details
acoustic = result['acoustic_summary']
print(f"Speaking rate: {acoustic['speaking_rate_syllables_per_sec']} syl/s")
print(f"Pitch: {acoustic['pitch_mean_hz']} Hz")

# Access annotated segments
for seg in result['segments']:
    print(f"{seg['label']}: {seg['start_ms']}-{seg['end_ms']}ms (weight={seg['weight']})")

# Full report
report = result['report']
```

### Method 2: REST API

**Start the server:**

```bash
poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Test with curl:**

```bash
# Health check
curl http://localhost:8000/healthz

# Analyze audio
curl -X POST http://localhost:8000/v1/speech/analyse \
  -F "audio_file=@patient_recording.wav" \
  -F "patient_age=72" \
  -F "onset_hours=3.2"
```

**Test with Python requests:**

```python
import requests

url = "http://localhost:8000/v1/speech/analyse"

with open("patient_recording.wav", "rb") as f:
    files = {"audio_file": f}
    data = {
        "patient_age": 72,
        "onset_hours": 3.2
    }

    response = requests.post(url, files=files, data=data)
    result = response.json()

print(f"Slurring Score: {result['slurring_score']}")
print(f"Risk Tier: {result['risk_tier']}")
```

**Interactive Testing:**

Open http://localhost:8000/docs in your browser for Swagger UI. You can:
- Upload files via web interface
- See request/response schemas
- Test all endpoints interactively

---

## Understanding the Code

### Key Design Patterns

#### 1. Pydantic Schemas for Data Validation

```python
# All data structures use Pydantic for automatic validation
from pydantic import BaseModel, Field

class SlurringResult(BaseModel):
    slurring_score: float = Field(..., ge=0, le=100)  # Must be 0-100
    severity: SeverityLevel  # Enum validation
    confidence: float = Field(..., ge=0, le=1)  # Must be 0-1

# Automatic validation on creation
result = SlurringResult(
    slurring_score=150,  # ❌ ValidationError: must be <= 100
    severity="unknown",  # ❌ ValidationError: not a valid SeverityLevel
    confidence=1.5       # ❌ ValidationError: must be <= 1
)
```

#### 2. Pipeline Pattern

```python
# Each module exposes a clean interface
class AudioPreprocessor:
    def process(self, waveform, sr, duration) -> PreprocessedAudio:
        # Input: raw waveform
        # Output: clean PreprocessedAudio object
        pass

# Pipeline chains these together
class SlurringDetectionPipeline:
    def analyse(self, audio_file):
        # Chain: load → preprocess → extract → predict → score → report
        audio = self.loader.load(audio_file)
        preprocessed = self.preprocessor.process(audio)
        features = self.extractor.extract(preprocessed)
        # ... etc
```

#### 3. Dependency Injection (FastAPI)

```python
# Dependencies are injected, not created globally
from fastapi import Depends

@lru_cache  # Singleton pattern
def get_pipeline() -> SlurringDetectionPipeline:
    return SlurringDetectionPipeline()

@router.post("/analyse")
async def analyse(
    pipeline: SlurringDetectionPipeline = Depends(get_pipeline)
):
    # pipeline is injected, shared across requests
    result = pipeline.analyse(...)
```

---

## Extending the System

### Adding a New Feature Extractor

```python
# 1. Create schema in src/features/schemas.py
class MyNewFeatures(BaseModel):
    feature_1: float
    feature_2: np.ndarray

# 2. Create extractor in src/features/my_new_extractor.py
class MyNewExtractor:
    def extract(self, waveform: np.ndarray, sr: int) -> MyNewFeatures:
        # Your extraction logic
        feature_1 = compute_something(waveform)
        feature_2 = compute_something_else(waveform)

        return MyNewFeatures(
            feature_1=feature_1,
            feature_2=feature_2
        )

# 3. Add to FeatureBundle schema
class FeatureBundle(BaseModel):
    # ... existing features ...
    my_new_features: MyNewFeatures  # Add this

# 4. Integrate in pipeline.py
self.my_new_extractor = MyNewExtractor()
# ... in analyse():
my_new_features = self.my_new_extractor.extract(waveform, sr)
feature_bundle.my_new_features = my_new_features

# 5. Update feature fusion if needed
# Add to FeatureFusion.fuse() to include in fused vector
```

### Replacing Placeholder Models

```python
# Current: Placeholder
class ModelRegistry:
    def __init__(self, use_placeholder: bool = True):
        self.use_placeholder = use_placeholder

# After training:
# 1. Train your models and save to MLflow
# 2. Update ModelRegistry to load from MLflow:

class ModelRegistry:
    def load_ensemble(self, version="latest"):
        if self.use_placeholder:
            return PlaceholderEnsemble()
        else:
            # Load from MLflow
            import mlflow
            model_uri = f"models:/ensemble_dysarthria/{version}"
            model = mlflow.pytorch.load_model(model_uri)
            return model

# 3. In pipeline.py, change:
self.model_registry = ModelRegistry(use_placeholder=False)
```

### Adding Real Explainability

```python
# Currently in pipeline.py:
def _generate_mock_segments(self, duration_sec):
    # Returns mock segments

# Replace with:
def _generate_real_segments(self, features, attention_weights, gradcam_output):
    from src.explainability.segment_localiser import SegmentLocaliser

    localiser = SegmentLocaliser()
    segments = localiser.localise(
        attention_scores=attention_weights,
        gradcam_heatmap=gradcam_output,
        hop_length=512,
        sr=16000
    )
    return segments
```

---

## Troubleshooting Guide

### Common Issues & Solutions

#### Issue: `ModuleNotFoundError: No module named 'src'`

**Cause:** Python can't find the src package.

**Solution:**
```bash
# Ensure you're in the project root
cd slurry_speech

# Install in editable mode
poetry install

# Or with pip
pip install -e .
```

#### Issue: `torch.cuda.OutOfMemoryError`

**Cause:** GPU memory exhausted (when you have real models).

**Solution:**
```python
# Use CPU instead
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Or in configs/inference_config.yaml:
device:
  type: cpu
```

#### Issue: OpenSMILE features are all zeros

**Cause:** OpenSMILE not installed or failing.

**Solution:**
```bash
# Install OpenSMILE
poetry add opensmile

# If still failing, check logs - falls back to placeholder zeros
```

#### Issue: Silero VAD fails, using fallback

**Cause:** torch.hub download issue or model incompatibility.

**Solution:**
- VAD automatically falls back to energy-based detection
- Check logs for details
- Ensure internet connection for first-time model download

#### Issue: Audio file "duration too short/long"

**Cause:** Audio outside 5-60 second range.

**Solution:**
```python
# Adjust in pipeline initialization:
pipeline = SlurringDetectionPipeline()
pipeline.audio_loader = AudioLoader(
    min_duration=3.0,   # Allow shorter audio
    max_duration=120.0  # Allow longer audio
)
```

#### Issue: API returns 500 error

**Cause:** Exception in pipeline.

**Solution:**
```bash
# Check API logs
poetry run uvicorn api.main:app --log-level debug

# Check for specific error in output
# Most common: audio quality validation failure
```

---

## Performance Optimization Tips

### For Development

```python
# Faster feature extraction (skip expensive features)
pipeline.egemaps_extractor = None  # Skip eGeMAPS
pipeline.formant_extractor = None  # Skip formants

# In analyse():
if self.egemaps_extractor:
    egemaps = self.egemaps_extractor.extract(...)
else:
    egemaps = EGeMAPSFeatures(features=np.zeros(88))
```

### For Production (After Training Real Models)

```yaml
# configs/inference_config.yaml
device:
  type: cuda  # Use GPU

precision:
  use_mixed_precision: true  # fp16 inference
  dtype: float16

model:
  compile: true  # PyTorch 2.0 compilation
  compile_mode: reduce-overhead
```

---

## Next Steps

1. **Test the system** with your own audio files
2. **Understand the data flow** by adding print statements
3. **Download the Kaggle dataset** to prepare for training
4. **Implement training scripts** to replace placeholder models
5. **Add real explainability** with Grad-CAM implementation
6. **Deploy to production** with Docker

---

## Summary

You now have:
- ✅ Complete end-to-end pipeline
- ✅ Real feature extraction
- ✅ Placeholder models (train to replace)
- ✅ Full scoring and risk assessment
- ✅ JSON report generation
- ✅ FastAPI service

**The system works!** It processes audio, extracts features, scores dysarthria, assesses stroke risk, and generates reports. Now train real models to get accurate predictions!

Questions? Check QUICKSTART.md for specific commands or consult the implementation in src/pipeline.py.
