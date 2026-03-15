#!/usr/bin/env python3
"""Test the API endpoint with trained model."""

import requests
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/v1/speech/analyse"

# Test audio file
audio_file = Path("data/raw/kaggle_dysarthria/Dysarthria and Non Dysarthria/Dataset/Male_dysarthria/M04/Session2/Wav/0298.wav")

if not audio_file.exists():
    print(f"❌ Audio file not found: {audio_file}")
    exit(1)

print("="*80)
print("  Testing API with Trained HuBERT Model")
print("="*80)
print(f"\n🎤 Audio file: {audio_file.name}")
print(f"📍 Endpoint: {API_URL}\n")

# Prepare request
files = {
    'audio_file': ('test_audio.wav', open(audio_file, 'rb'), 'audio/wav')
}
data = {
    'patient_age': 65,
    'onset_hours': 2.5
}

print("📤 Sending request...")

try:
    response = requests.post(API_URL, files=files, data=data, timeout=60)

    print(f"✓ Response received (status: {response.status_code})\n")

    if response.status_code == 200:
        result = response.json()

        print("="*80)
        print("  API RESPONSE")
        print("="*80)
        print(f"Request ID: {result.get('request_id', 'N/A')}")
        print(f"Model Version: {result.get('model_version', 'N/A')}")
        print(f"\n📊 Results:")
        print(f"  Slurring Score: {result.get('slurring_score', 0):.1f}/100")
        print(f"  Severity: {result.get('severity', 'N/A')}")
        print(f"  Risk Score: {result.get('risk_score', 0):.1f}/100")
        print(f"  Risk Tier: {result.get('risk_tier', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0):.2%}")
        print(f"  Processing Time: {result.get('processing_time_ms', 0):.0f}ms")

        if result.get('emergency_alert'):
            print(f"\n⚠️  EMERGENCY ALERT: {result.get('emergency_alert')}")

        print(f"\nReport URL: {result.get('report_url', 'N/A')}")
        print("="*80)

        print("\n✅ API test successful! The trained model is working via API.")

    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
    print("\nMake sure the API server is running:")
    print("  python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
