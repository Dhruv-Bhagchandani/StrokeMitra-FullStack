#!/usr/bin/env python3
"""
Quick test script for Speech Slurring Detection System.

Usage:
    # Option 1: Use your own audio file
    python test_voice_input.py --audio path/to/your/audio.wav

    # Option 2: Generate synthetic test audio
    python test_voice_input.py --generate

    # Option 3: Record your voice (requires microphone)
    python test_voice_input.py --record --duration 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import soundfile as sf
from datetime import datetime
import json

from src.pipeline import SlurringDetectionPipeline


def generate_synthetic_audio(output_path: Path, duration: float = 5.0, sample_rate: int = 16000):
    """Generate synthetic audio with varying frequency (simulates speech-like signal)."""
    print(f"🎵 Generating synthetic audio ({duration}s at {sample_rate}Hz)...")

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a complex waveform that mimics speech patterns
    # Mix of frequencies with amplitude modulation
    fundamental = 150  # Hz (typical human speech F0)

    audio = (
        0.3 * np.sin(2 * np.pi * fundamental * t) +  # Fundamental frequency
        0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * fundamental * 3 * t) +  # Second harmonic
        0.05 * np.random.randn(len(t))  # Add some noise
    )

    # Amplitude modulation to simulate syllables
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
    audio = audio * modulation

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    # Save
    sf.write(output_path, audio, sample_rate)
    print(f"✓ Synthetic audio saved to: {output_path}")
    return output_path


def record_audio(output_path: Path, duration: float = 5.0, sample_rate: int = 16000):
    """Record audio from microphone."""
    try:
        import sounddevice as sd

        print(f"🎤 Recording audio for {duration} seconds...")
        print("   Speak now!")

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        print("✓ Recording complete!")

        # Save
        sf.write(output_path, audio, sample_rate)
        print(f"✓ Audio saved to: {output_path}")
        return output_path

    except ImportError:
        print("❌ sounddevice not installed. Install with: pip install sounddevice")
        print("   Falling back to synthetic audio...")
        return generate_synthetic_audio(output_path, duration, sample_rate)
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        print("   Falling back to synthetic audio...")
        return generate_synthetic_audio(output_path, duration, sample_rate)


def analyze_audio(audio_path: Path, patient_age: int = 65, onset_hours: float = 2.0):
    """Run the slurring detection pipeline on audio."""
    print("\n" + "=" * 80)
    print("  SPEECH SLURRING DETECTION - ANALYSIS")
    print("=" * 80)

    print(f"\n📂 Input Audio: {audio_path}")
    print(f"👤 Patient Age: {patient_age}")
    print(f"⏰ Onset Hours: {onset_hours}")

    # Initialize pipeline
    print("\n⚙️  Initializing pipeline...")
    pipeline = SlurringDetectionPipeline()

    # Run analysis
    print("🔬 Running analysis (this may take a moment)...")
    try:
        result = pipeline.analyse(
            audio_file=audio_path,
            patient_age=patient_age,
            onset_hours=onset_hours,
        )

        # Display results
        print("\n" + "=" * 80)
        print("  ANALYSIS RESULTS")
        print("=" * 80)

        print(f"\n📊 SLURRING ASSESSMENT:")
        print(f"   Score:      {result['slurring_score']:.1f} / 100")
        print(f"   Severity:   {result['severity'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")

        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"   Risk Score: {result['risk_score']:.1f} / 100")
        print(f"   Risk Tier:  {result['risk_tier'].upper()}")

        if result.get('emergency_alert'):
            print(f"\n🚨 EMERGENCY ALERT: CRITICAL RISK DETECTED!")

        print(f"\n🔍 ACOUSTIC SUMMARY:")
        acoustic = result.get('acoustic_summary', {})
        if acoustic:
            print(f"   F0 Mean:        {acoustic.get('f0_mean', 0):.1f} Hz")
            print(f"   Speaking Rate:  {acoustic.get('speaking_rate', 0):.2f} syllables/s")
            print(f"   Pause Ratio:    {acoustic.get('pause_ratio', 0):.2%}")

        print(f"\n📈 SEGMENTS:")
        segments = result.get('segments', [])
        if segments:
            print(f"   {len(segments)} segment(s) analyzed")
            for i, seg in enumerate(segments[:3], 1):  # Show first 3
                print(f"   {i}. {seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s: "
                      f"{seg.get('severity', 'unknown').upper()} (confidence: {seg.get('confidence', 0):.2%})")
        else:
            print("   No segments detected (this is normal with placeholder models)")

        print(f"\n📄 REPORT:")
        if result.get('report_json'):
            print(f"   JSON Report: Available in result")
        if result.get('report_pdf_base64'):
            print(f"   PDF Report:  Generated (base64 encoded)")

        print(f"\n⏱️  PROCESSING:")
        print(f"   Time:    {result.get('processing_time_ms', 0):.0f} ms")
        print(f"   Model:   {result.get('model_version', 'unknown')}")

        print("\n" + "=" * 80)

        # Save full result to JSON
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_output = output_dir / f"result_{timestamp}.json"

        # Remove binary data before saving
        result_to_save = {k: v for k, v in result.items() if k != 'report_pdf_base64'}

        with open(json_output, 'w') as f:
            json.dump(result_to_save, f, indent=2, default=str)

        print(f"\n✓ Full results saved to: {json_output}")

        # Save PDF if available
        if result.get('report_pdf_base64'):
            import base64
            pdf_output = output_dir / f"report_{timestamp}.pdf"
            pdf_bytes = base64.b64decode(result['report_pdf_base64'])
            with open(pdf_output, 'wb') as f:
                f.write(pdf_bytes)
            print(f"✓ PDF report saved to: {pdf_output}")

        print("\n✓ Analysis complete!")

        return result

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Speech Slurring Detection System")

    # Input options
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--generate', action='store_true', help='Generate synthetic test audio')
    parser.add_argument('--record', action='store_true', help='Record audio from microphone')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration for recording/generation (seconds)')

    # Analysis options
    parser.add_argument('--age', type=int, default=65, help='Patient age')
    parser.add_argument('--onset', type=float, default=2.0, help='Symptom onset (hours)')

    args = parser.parse_args()

    # Determine audio source
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            return
    elif args.record:
        test_dir = Path("test_outputs")
        test_dir.mkdir(exist_ok=True)
        audio_path = test_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        audio_path = record_audio(audio_path, duration=args.duration)
    elif args.generate:
        test_dir = Path("test_outputs")
        test_dir.mkdir(exist_ok=True)
        audio_path = test_dir / "synthetic_test.wav"
        audio_path = generate_synthetic_audio(audio_path, duration=args.duration)
    else:
        # Default: generate synthetic
        print("No audio source specified. Generating synthetic test audio...")
        test_dir = Path("test_outputs")
        test_dir.mkdir(exist_ok=True)
        audio_path = test_dir / "synthetic_test.wav"
        audio_path = generate_synthetic_audio(audio_path, duration=args.duration)

    # Analyze
    analyze_audio(audio_path, patient_age=args.age, onset_hours=args.onset)


if __name__ == "__main__":
    main()
