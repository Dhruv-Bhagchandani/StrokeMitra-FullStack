#!/usr/bin/env python3
"""
Simple demonstration of the Speech Slurring Detection System architecture.

This demonstrates the flow without requiring ML dependencies.
Shows what each component does and the expected output structure.
"""

import json
from datetime import datetime


def demo_pipeline_flow():
    """Demonstrate the complete pipeline flow with mock data."""

    print("\n" + "=" * 80)
    print("  SPEECH SLURRING DETECTION SYSTEM - ARCHITECTURE DEMO")
    print("=" * 80)

    # Simulated audio file input
    audio_file = "patient_speech_sample.wav"
    patient_age = 65
    onset_hours = 2.5

    print(f"\n📂 INPUT:")
    print(f"   Audio File:   {audio_file}")
    print(f"   Patient Age:  {patient_age} years")
    print(f"   Onset Time:   {onset_hours} hours ago")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Audio Ingestion (SF-01)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n" + "─" * 80)
    print("STEP 1: Audio Ingestion & Preprocessing (SF-01)")
    print("─" * 80)
    print("✓ Load audio file (supports wav/mp3/ogg/m4a)")
    print("✓ Resample to 16kHz")
    print("✓ Normalize loudness (ITU-R BS.1770)")
    print("✓ Trim silence (VAD)")
    print("✓ Quality check (SNR, clipping detection)")

    preprocessed_audio = {
        "sample_rate": 16000,
        "duration_sec": 8.5,
        "channels": 1,
        "snr_db": 25.3,
        "clipping_ratio": 0.002,
        "is_valid": True
    }
    print(f"\nPreprocessed Audio:")
    for key, value in preprocessed_audio.items():
        print(f"   {key:20s}: {value}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Feature Extraction (SF-02)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n" + "─" * 80)
    print("STEP 2: Acoustic Feature Extraction (SF-02)")
    print("─" * 80)

    print("\n✓ MFCC Features (39-dim):")
    print("   - 13 MFCCs + delta + delta-delta")
    print("   - Captures spectral envelope")

    print("\n✓ Prosodic Features:")
    print("   - F0 (pitch): 142.5 Hz (mean), 28.3 Hz (std)")
    print("   - Energy: 0.42 (mean)")
    print("   - Speaking rate: 3.2 syllables/sec")
    print("   - Pause ratio: 18.5%")

    print("\n✓ Formant Features:")
    print("   - F1 (first formant): 680 Hz")
    print("   - F2 (second formant): 1520 Hz")
    print("   - F3 (third formant): 2450 Hz")
    print("   - Vowel space area: 142,000 Hz²")

    print("\n✓ eGeMAPS Features (88-dim):")
    print("   - OpenSMILE extended Geneva Minimalistic Acoustic Parameter Set")

    print("\n✓ Spectrogram Features:")
    print("   - Log-mel spectrogram (128 mel bins)")
    print("   - CWT scalogram (wavelet transform)")

    feature_bundle = {
        "mfcc_dims": 39,
        "prosody_dims": 8,
        "formant_dims": 7,
        "egemaps_dims": 88,
        "spectrogram_shape": [2, 128, 680],  # [channels, freq, time]
        "fused_acoustic_dims": 145
    }

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Model Inference (SF-03)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n" + "─" * 80)
    print("STEP 3: Dual-Model Ensemble Inference (SF-03)")
    print("─" * 80)

    print("\n✓ Branch 1: HuBERT-SALR")
    print("   - Input: Raw waveform")
    print("   - Architecture: HuBERT-large (24 layers, 1024-dim)")
    print("   - Layer-weighted pooling across all transformer layers")
    print("   - SALR head: Classification + Embedding for triplet loss")
    print("   - Output logits: [0.2, 0.8] → 80% dysarthric")

    print("\n✓ Branch 2: CNN-BiLSTM-Transformer")
    print("   - Input: Log-mel + CWT spectrogram")
    print("   - CNN: ResNet-style blocks for spatial features")
    print("   - BiLSTM: Temporal modeling (2 layers, 256 hidden)")
    print("   - Transformer: Self-attention (8 heads, 4 layers)")
    print("   - Output logits: [0.35, 0.65] → 65% dysarthric")

    print("\n✓ Ensemble Fusion:")
    print("   - Alpha (HuBERT weight): 0.6")
    print("   - Ensemble logits = 0.6 * [0.2, 0.8] + 0.4 * [0.35, 0.65]")
    print("   - Final logits: [0.26, 0.74] → 74% dysarthric")

    print("\n✓ Calibration (Platt Scaling):")
    print("   - Raw probability: 0.74")
    print("   - Calibrated probability: 0.68 (sigmoid(a*logit + b))")

    model_output = {
        "raw_probability": 0.74,
        "calibrated_probability": 0.68,
        "confidence": 0.87,
        "model_version": "ensemble-v1.0"
    }

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Slurring Score (SF-04)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n" + "─" * 80)
    print("STEP 4: Slurring Score Computation (SF-04)")
    print("─" * 80)

    slurring_score = model_output["calibrated_probability"] * 100
    print(f"\n✓ Slurring Score: {slurring_score:.1f} / 100")

    # Severity classification
    if slurring_score < 20:
        severity = "none"
    elif slurring_score < 45:
        severity = "mild"
    elif slurring_score < 70:
        severity = "moderate"
    else:
        severity = "severe"

    print(f"✓ Severity: {severity.upper()}")
    print(f"   Thresholds: 0-20 (None), 21-45 (Mild), 46-70 (Moderate), 71-100 (Severe)")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Explainability (SF-05)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n" + "─" * 80)
    print("STEP 5: Explainability Analysis (SF-05)")
    print("─" * 80)

    print("\n✓ Grad-CAM Heatmap:")
    print("   - Highlights time-frequency regions triggering detection")
    print("   - Peak activation at 2.1s-3.5s (moderate slurring)")

    print("\n✓ Attention Rollout:")
    print("   - Transformer attention weights across layers")
    print("   - High attention at 4.2s-5.8s (vowel prolongation)")

    print("\n✓ Segment Localization:")
    segments = [
        {"start": 0.0, "end": 2.0, "severity": "none", "confidence": 0.92},
        {"start": 2.1, "end": 3.5, "severity": "moderate", "confidence": 0.85},
        {"start": 3.6, "end": 5.8, "severity": "mild", "confidence": 0.78},
        {"start": 5.9, "end": 8.5, "severity": "moderate", "confidence": 0.81},
    ]

    for i, seg in enumerate(segments, 1):
        print(f"   Segment {i}: {seg['start']:.1f}s - {seg['end']:.1f}s")
        print(f"      Severity: {seg['severity'].upper()} (confidence: {seg['confidence']:.2%})")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Risk Assessment (SF-06)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n" + "─" * 80)
    print("STEP 6: Clinical Risk Assessment (SF-06)")
    print("─" * 80)

    # Age factor (normalized to 0-1)
    age_factor = (patient_age - 40) / (80 - 40)  # (65-40)/(80-40) = 0.625

    # Onset factor (golden window: 4.5 hours)
    onset_factor = 1.0 if onset_hours <= 4.5 else 0.5  # Within golden window

    print(f"\n✓ Risk Factors:")
    print(f"   Slurring Score: {slurring_score:.1f}")
    print(f"   Age Factor:     {age_factor:.3f} (patient age: {patient_age})")
    print(f"   Onset Factor:   {onset_factor:.3f} ({'within' if onset_factor == 1.0 else 'outside'} golden window)")

    # Logistic regression
    w0, w1, w2, w3 = -2.0, 0.05, 1.5, 0.8  # Weights from config
    logit = w0 + w1 * slurring_score + w2 * age_factor + w3 * onset_factor

    # Sigmoid
    import math
    risk_score = (1 / (1 + math.exp(-logit))) * 100

    print(f"\n✓ Risk Score: {risk_score:.1f} / 100")

    # Risk tier
    if risk_score < 25:
        risk_tier = "low"
    elif risk_score < 50:
        risk_tier = "moderate"
    elif risk_score < 75:
        risk_tier = "high"
    else:
        risk_tier = "critical"

    print(f"✓ Risk Tier: {risk_tier.upper()}")

    emergency_alert = (risk_tier == "critical")
    if emergency_alert:
        print(f"\n🚨 EMERGENCY ALERT TRIGGERED!")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Report Generation (SF-07)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n" + "─" * 80)
    print("STEP 7: Clinical Report Generation (SF-07)")
    print("─" * 80)

    print("\n✓ JSON Report:")
    print("   - Machine-readable format")
    print("   - All metrics, segments, and acoustic features")
    print("   - Ready for EHR integration")

    print("\n✓ PDF Report:")
    print("   - Clinical-grade formatting")
    print("   - Color-coded severity indicators")
    print("   - Waveform and spectrogram visualizations")
    print("   - Segment annotations")
    print("   - Recommendation section")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n" + "=" * 80)
    print("  FINAL ANALYSIS RESULTS")
    print("=" * 80)

    result = {
        "request_id": "test-demo-001",
        "timestamp": datetime.utcnow().isoformat(),
        "slurring_score": round(slurring_score, 1),
        "severity": severity,
        "confidence": model_output["confidence"],
        "risk_score": round(risk_score, 1),
        "risk_tier": risk_tier,
        "emergency_alert": emergency_alert,
        "acoustic_summary": {
            "f0_mean": 142.5,
            "energy_mean": 0.42,
            "speaking_rate": 3.2,
            "pause_ratio": 0.185,
            "vowel_space_area": 142000,
        },
        "segments": segments,
        "model_version": model_output["model_version"],
        "processing_time_ms": 1240,
    }

    print(f"\n{json.dumps(result, indent=2)}")

    print("\n" + "=" * 80)
    print("  SYSTEM COMPONENTS DEMONSTRATED")
    print("=" * 80)
    print("\n✓ SF-01: Audio Ingestion & Preprocessing")
    print("✓ SF-02: Multi-modal Feature Extraction")
    print("✓ SF-03: Dual-Model Ensemble (HuBERT-SALR + CNN-BiLSTM)")
    print("✓ SF-04: Slurring Score & Severity Classification")
    print("✓ SF-05: Explainability (Grad-CAM, Attention, Segments)")
    print("✓ SF-06: Clinical Risk Assessment")
    print("✓ SF-07: JSON & PDF Report Generation")

    print("\n" + "=" * 80)
    print("\n💡 NOTE: This demo uses mock data to show the architecture.")
    print("   With real trained models, predictions will be based on actual audio analysis.")
    print("\n🚀 To run with real audio:")
    print("   1. Install dependencies: poetry install")
    print("   2. Download dataset: python scripts/download_dataset.py")
    print("   3. Train models: bash scripts/run_training.sh")
    print("   4. Test with audio: python test_voice_input.py --audio your_file.wav")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    demo_pipeline_flow()
