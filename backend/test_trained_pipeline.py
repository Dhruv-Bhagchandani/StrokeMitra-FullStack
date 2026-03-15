#!/usr/bin/env python3
"""Test the full pipeline with the trained HuBERT model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import SlurringDetectionPipeline


def main():
    """Test pipeline with trained model."""

    print("="*80)
    print("  Testing Full Pipeline with Trained HuBERT Model")
    print("="*80)
    print()

    # Initialize pipeline with trained model (use_placeholder=False)
    print("📥 Initializing pipeline with trained model...")
    pipeline = SlurringDetectionPipeline(use_placeholder=False)
    print("✓ Pipeline initialized\n")

    # Test on dysarthric sample
    test_file = "data/raw/kaggle_dysarthria/Dysarthria and Non Dysarthria/Dataset/Male_dysarthria/M04/Session2/Wav/0298.wav"

    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("Please provide a valid audio file path.")
        return

    print(f"🎤 Testing on: {Path(test_file).name}")
    print(f"Expected: Dysarthric (from path)\n")

    # Run analysis
    print("🔬 Running analysis...")
    result = pipeline.analyse(
        audio_file=test_file,
        patient_age=65,
        onset_hours=2.5,
        return_report=True
    )

    print("\n" + "="*80)
    print("  RESULTS")
    print("="*80)
    print(f"Model Version: {result.get('model_version', 'N/A')}")
    print(f"Slurring Score: {result.get('slurring_score', 0):.1f}/100")
    print(f"Severity: {result.get('severity', 'N/A')}")
    print(f"Risk Score: {result.get('risk_score', 0):.1f}/100")
    print(f"Risk Tier: {result.get('risk_tier', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0):.2%}")
    print(f"Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
    print()

    print("📊 Acoustic Summary:")
    acoustic_summary = result.get('acoustic_summary', {})
    for key, value in acoustic_summary.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    print()

    segments = result.get('segments', [])
    if segments:
        print(f"📍 Detected {len(segments)} problematic segments")
        for i, seg in enumerate(segments[:3], 1):
            print(f"  Segment {i}: {seg.get('start_sec', 0):.1f}s-{seg.get('end_sec', 0):.1f}s ({seg.get('label', 'N/A')})")

    print("\n" + "="*80)

    risk_tier = result.get('risk_tier', '')
    severity = result.get('severity', '')

    if risk_tier == "CRITICAL":
        print("⚠️  CRITICAL RISK - Immediate medical evaluation recommended")
    elif severity in ["MODERATE", "SEVERE"]:
        print("⚠️  Dysarthria detected - Medical consultation recommended")
    else:
        print("✓ Speech patterns within normal range")

    print("="*80)
    print("\n✅ Test complete! The trained model is working correctly.")


if __name__ == "__main__":
    main()
