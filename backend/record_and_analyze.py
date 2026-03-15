#!/usr/bin/env python3
"""Record audio from microphone and analyze for dysarthria."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
from datetime import datetime
import tempfile

from src.pipeline import SlurringDetectionPipeline


def record_audio(duration=10, sample_rate=16000):
    """
    Record audio from microphone.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate (16000 for compatibility)

    Returns:
        Path to saved audio file
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ sounddevice not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice", "--quiet"])
        import sounddevice as sd

    print("\n" + "="*80)
    print("  🎤 MICROPHONE RECORDING")
    print("="*80)
    print(f"\n⏱️  Recording for {duration} seconds...")
    print("🔴 RECORDING NOW - Start speaking!\n")

    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is finished

    print("✅ Recording complete!\n")

    # Save to temporary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = tempfile.gettempdir() + f"/recording_{timestamp}.wav"
    sf.write(temp_file, recording, sample_rate)

    print(f"💾 Saved to: {temp_file}")

    return temp_file


def analyze_recording(audio_file, patient_age=None, onset_hours=None):
    """
    Analyze recorded audio.

    Args:
        audio_file: Path to audio file
        patient_age: Patient age (optional)
        onset_hours: Hours since symptom onset (optional)
    """
    print("\n" + "="*80)
    print("  🔬 ANALYZING YOUR SPEECH")
    print("="*80)
    print("\n📥 Loading trained model...")

    # Initialize pipeline with trained model
    pipeline = SlurringDetectionPipeline(use_placeholder=False)

    print("✓ Model loaded\n")
    print("🧠 Processing audio (this may take a few seconds)...\n")

    # Analyze
    result = pipeline.analyse(
        audio_file=audio_file,
        patient_age=patient_age,
        onset_hours=onset_hours,
        return_report=True
    )

    # Display results
    print("\n" + "="*80)
    print("  📊 YOUR RESULTS")
    print("="*80)

    slurring_score = result.get('slurring_score', 0)
    severity = result.get('severity', 'N/A').upper()
    risk_score = result.get('risk_score', 0)
    risk_tier = result.get('risk_tier', 'N/A').upper()
    confidence = result.get('confidence', 0)

    print(f"\n🎯 Slurring Score: {slurring_score:.1f}/100")
    print(f"   Severity: {severity}")
    print(f"   Confidence: {confidence:.1%}\n")

    print(f"⚠️  Risk Score: {risk_score:.1f}/100")
    print(f"   Risk Tier: {risk_tier}\n")

    # Interpretation
    print("-" * 80)
    print("📋 INTERPRETATION:")
    print("-" * 80)

    if severity == "NONE":
        print("✅ No signs of dysarthria detected")
        print("   Your speech patterns appear normal")
    elif severity == "MILD":
        print("⚠️  Mild dysarthria detected")
        print("   Subtle changes in speech articulation noticed")
    elif severity == "MODERATE":
        print("⚠️  Moderate dysarthria detected")
        print("   Noticeable speech impairment present")
    elif severity == "SEVERE":
        print("🚨 SEVERE dysarthria detected")
        print("   Significant speech impairment identified")

    print()

    if risk_tier == "CRITICAL":
        print("🚨🚨🚨 CRITICAL RISK - SEEK IMMEDIATE MEDICAL ATTENTION 🚨🚨🚨")
        print("   This suggests possible acute neurological symptoms")
        print("   Call emergency services or go to the nearest ER NOW")
    elif risk_tier == "HIGH":
        print("⚠️  HIGH RISK - Urgent medical evaluation recommended")
        print("   Contact your doctor or visit urgent care today")
    elif risk_tier == "MODERATE":
        print("⚠️  MODERATE RISK - Medical assessment recommended")
        print("   Schedule an appointment with your healthcare provider")
    else:
        print("✅ LOW RISK - Continue monitoring")
        print("   Speech patterns within normal range")

    print()

    # Acoustic summary
    acoustic = result.get('acoustic_summary', {})
    if acoustic:
        print("-" * 80)
        print("🔊 ACOUSTIC ANALYSIS:")
        print("-" * 80)
        print(f"   Speaking Rate: {acoustic.get('speaking_rate_syllables_per_sec', 0):.2f} syllables/sec")
        print(f"   Pitch (Mean): {acoustic.get('pitch_mean_hz', 0):.1f} Hz")
        print(f"   Pitch Variability: {acoustic.get('pitch_variability_hz', 0):.1f} Hz")
        print(f"   Pause Ratio: {acoustic.get('pause_ratio', 0):.2%}")
        print(f"   Vowel Space Area: {acoustic.get('vowel_space_area', 0):.0f} Hz²")

    print("\n" + "="*80)
    print(f"⏱️  Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
    print(f"🤖 Model Version: {result.get('model_version', 'N/A')}")
    print("="*80)

    return result


def main():
    """Main function."""
    print("\n" + "="*80)
    print("  🎤 REAL-TIME SPEECH ANALYSIS FOR DYSARTHRIA DETECTION")
    print("="*80)
    print("\nThis system will:")
    print("  1. Record your voice for 10 seconds")
    print("  2. Analyze speech patterns using AI")
    print("  3. Provide dysarthria risk assessment")
    print("\n⚕️  MEDICAL DISCLAIMER:")
    print("  This is a screening tool, NOT a diagnostic device.")
    print("  Always consult healthcare professionals for medical advice.")
    print("\n" + "="*80)

    # Get optional patient info
    try:
        age_input = input("\n📝 Enter your age (or press Enter to skip): ").strip()
        patient_age = int(age_input) if age_input else None
    except ValueError:
        patient_age = None

    try:
        onset_input = input("📝 Hours since symptom onset (or press Enter to skip): ").strip()
        onset_hours = float(onset_input) if onset_input else None
    except ValueError:
        onset_hours = None

    input("\n✋ Press ENTER when ready to start recording...")

    try:
        # Record audio
        audio_file = record_audio(duration=10)

        # Analyze
        result = analyze_recording(audio_file, patient_age, onset_hours)

        print("\n✅ Analysis complete!")
        print(f"\n💾 Your recording is saved at: {audio_file}")
        print("   You can re-analyze it later or share with your doctor.\n")

    except KeyboardInterrupt:
        print("\n\n❌ Recording cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure your microphone is connected")
        print("  2. Grant microphone permissions if prompted")
        print("  3. Try running: pip3 install sounddevice")
        sys.exit(1)


if __name__ == "__main__":
    main()
