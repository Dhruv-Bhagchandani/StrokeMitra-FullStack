"""
Main pipeline orchestrator for speech slurring detection.

This module chains all sub-features (SF-01 through SF-07) into an end-to-end pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.ingestion.audio_loader import AudioLoader
from src.ingestion.preprocessor import AudioPreprocessor
from src.ingestion.vad import VoiceActivityDetector
from src.ingestion.quality_checker import QualityChecker

from src.features.mfcc_extractor import MFCCExtractor
from src.features.prosodic_extractor import ProsodicExtractor
from src.features.formant_extractor import FormantExtractor
from src.features.egemaps_extractor import EGeMAPSExtractor
from src.features.spectrogram_builder import SpectrogramBuilder
from src.features.feature_fusion import FeatureFusion
from src.features.schemas import FeatureBundle

from src.models.model_registry import ModelRegistry
from src.models.ensemble import EnsembleModel
from src.models.calibration import PlattScaling

from src.scoring.slurring_scorer import SlurringScorer
from src.risk.risk_scorer import RiskScorer

from src.reporting.report_builder import ReportBuilder
from src.reporting.json_renderer import JSONRenderer
from src.reporting.schemas import SegmentAnnotation

logger = logging.getLogger(__name__)


class SlurringDetectionPipeline:
    """
    End-to-end pipeline for speech slurring detection.

    Chains:
        SF-01: Audio Ingestion
        SF-02: Feature Extraction
        SF-03: Model Inference
        SF-04: Slurring Scoring
        SF-05: Explainability (simplified in MVP)
        SF-06: Risk Assessment
        SF-07: Report Generation
    """

    def __init__(self, config_dir: str | Path = "configs/", use_placeholder: bool = True):
        """
        Initialize pipeline with all components.

        Args:
            config_dir: Path to configuration directory
            use_placeholder: If False, load trained model instead of placeholder
        """
        logger.info(f"Initializing SlurringDetectionPipeline (placeholder={use_placeholder})...")

        # SF-01: Audio Ingestion
        self.audio_loader = AudioLoader(max_duration=60.0, min_duration=5.0)
        self.preprocessor = AudioPreprocessor(target_sr=16000)
        self.vad = VoiceActivityDetector(threshold=0.5, sampling_rate=16000)
        self.quality_checker = QualityChecker()

        # SF-02: Feature Extraction
        self.mfcc_extractor = MFCCExtractor()
        self.prosodic_extractor = ProsodicExtractor()
        self.formant_extractor = FormantExtractor()
        self.egemaps_extractor = EGeMAPSExtractor()
        self.spectrogram_builder = SpectrogramBuilder()
        self.feature_fusion = FeatureFusion()

        # SF-03: Model Inference
        self.model_registry = ModelRegistry(use_placeholder=use_placeholder)
        self.ensemble_model = self.model_registry.load_ensemble()
        self.calibration = PlattScaling.identity()

        # SF-04, SF-06: Scoring
        self.slurring_scorer = SlurringScorer()
        self.risk_scorer = RiskScorer()

        # SF-07: Reporting
        self.report_builder = ReportBuilder()
        self.json_renderer = JSONRenderer()

        logger.info("✓ Pipeline initialized successfully")

    def analyse(
        self,
        audio_file: str | Path,
        patient_age: Optional[int] = None,
        onset_hours: Optional[float] = None,
        return_report: bool = True,
    ) -> dict:
        """
        Run end-to-end analysis on audio file.

        Args:
            audio_file: Path to audio file
            patient_age: Patient age (optional, improves risk scoring)
            onset_hours: Hours since symptom onset (optional)
            return_report: Whether to generate JSON report

        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()

        logger.info(f"=" * 80)
        logger.info(f"Starting analysis: {audio_file}")
        logger.info(f"=" * 80)

        # ──────────────────────────────────────────────────────────────────────
        # SF-01: Audio Ingestion & Preprocessing
        # ──────────────────────────────────────────────────────────────────────
        logger.info("SF-01: Audio Ingestion & Preprocessing")

        # Load audio
        audio_input, waveform = self.audio_loader.load(audio_file)

        # Check quality
        quality_metrics = self.quality_checker.check(
            waveform, audio_input.sample_rate, audio_input.duration_sec
        )

        if not quality_metrics.is_valid:
            raise ValueError(f"Audio quality validation failed: {quality_metrics.get_quality_summary()}")

        # Preprocess
        preprocessed = self.preprocessor.process(
            waveform, audio_input.sample_rate, audio_input.duration_sec
        )

        # Apply VAD
        speech_segments, speech_ratio = self.vad.detect_speech(preprocessed.waveform)
        preprocessed.speech_segments = speech_segments
        preprocessed.speech_ratio = speech_ratio
        preprocessed.vad_applied = True

        logger.info(f"✓ Preprocessed: {preprocessed.duration_sec:.2f}s, speech_ratio={speech_ratio:.2%}")

        # ──────────────────────────────────────────────────────────────────────
        # SF-02: Feature Extraction
        # ──────────────────────────────────────────────────────────────────────
        logger.info("SF-02: Feature Extraction")

        waveform = preprocessed.waveform
        sr = preprocessed.sample_rate

        # Extract all features
        mfcc_features = self.mfcc_extractor.extract(waveform, sr)
        prosodic_features = self.prosodic_extractor.extract(waveform, sr)
        formant_features = self.formant_extractor.extract(waveform, sr)
        egemaps_features = self.egemaps_extractor.extract(waveform, sr)
        spectrogram_features = self.spectrogram_builder.build(waveform, sr)

        # Create feature bundle
        feature_bundle = FeatureBundle(
            waveform=waveform,
            sample_rate=sr,
            duration_sec=preprocessed.duration_sec,
            mfcc=mfcc_features,
            prosody=prosodic_features,
            formants=formant_features,
            egemaps=egemaps_features,
            spectrogram=spectrogram_features,
        )

        # Fuse features
        feature_bundle = self.feature_fusion.fuse(feature_bundle)

        logger.info(f"✓ Features extracted: fused_acoustic={feature_bundle.fused_acoustic.shape}")

        # ──────────────────────────────────────────────────────────────────────
        # SF-03: Model Inference
        # ──────────────────────────────────────────────────────────────────────
        logger.info("SF-03: Model Inference (placeholder)")

        prediction = self.ensemble_model.predict(
            waveform=feature_bundle.waveform,
            spectrogram=feature_bundle.spectrogram.stacked,
            acoustic_features=feature_bundle.fused_acoustic,
        )

        raw_probability = prediction["raw_probability"]
        calibrated_probability = self.calibration.transform(raw_probability)
        confidence = float(np.max(prediction["probabilities"]))

        logger.info(f"✓ Model prediction: prob={calibrated_probability:.3f}, confidence={confidence:.3f}")

        # ──────────────────────────────────────────────────────────────────────
        # SF-04: Slurring Scoring
        # ──────────────────────────────────────────────────────────────────────
        logger.info("SF-04: Slurring Scoring")

        slurring_result = self.slurring_scorer.compute_score(
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            confidence=confidence,
            model_version=self.ensemble_model.version,
        )

        logger.info(
            f"✓ Slurring score: {slurring_result.slurring_score} "
            f"(severity={slurring_result.severity.value})"
        )

        # ──────────────────────────────────────────────────────────────────────
        # SF-05: Explainability (simplified - mock segments for MVP)
        # ──────────────────────────────────────────────────────────────────────
        logger.info("SF-05: Explainability (simplified)")

        segments = self._generate_mock_segments(preprocessed.duration_sec)

        logger.info(f"✓ Generated {len(segments)} annotated segments")

        # ──────────────────────────────────────────────────────────────────────
        # SF-06: Risk Assessment
        # ──────────────────────────────────────────────────────────────────────
        logger.info("SF-06: Risk Assessment")

        risk_assessment = self.risk_scorer.compute_risk(
            slurring_score=slurring_result.slurring_score,
            patient_age=patient_age,
            onset_hours=onset_hours,
        )

        logger.info(
            f"✓ Risk score: {risk_assessment.risk_score} "
            f"(tier={risk_assessment.risk_tier.value}, emergency={risk_assessment.emergency_alert})"
        )

        # ──────────────────────────────────────────────────────────────────────
        # SF-07: Report Generation
        # ──────────────────────────────────────────────────────────────────────
        logger.info("SF-07: Report Generation")

        processing_time_ms = (time.time() - start_time) * 1000
        acoustic_summary = feature_bundle.get_acoustic_summary()

        report_data = self.report_builder.build(
            slurring_result=slurring_result,
            risk_assessment=risk_assessment,
            acoustic_summary=acoustic_summary,
            segments=segments,
            processing_time_ms=processing_time_ms,
            audio_duration_sec=preprocessed.duration_sec,
            patient_age=patient_age,
            onset_hours=onset_hours,
        )

        report_output = None
        if return_report:
            report_output = self.json_renderer.render(report_data)
            logger.info(f"✓ Report generated: {report_data.report_id}")

        # ──────────────────────────────────────────────────────────────────────
        # Return results
        # ──────────────────────────────────────────────────────────────────────
        logger.info(f"=" * 80)
        logger.info(f"✓ Analysis complete: {processing_time_ms:.0f}ms")
        logger.info(f"=" * 80)

        return {
            "request_id": report_data.report_id,
            "slurring_score": slurring_result.slurring_score,
            "severity": slurring_result.severity.value,
            "risk_score": risk_assessment.risk_score,
            "risk_tier": risk_assessment.risk_tier.value,
            "confidence": confidence,
            "segments": [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "label": seg.label,
                    "weight": seg.weight,
                }
                for seg in segments
            ],
            "acoustic_summary": acoustic_summary,
            "processing_time_ms": processing_time_ms,
            "model_version": self.ensemble_model.version,
            "emergency_alert": risk_assessment.emergency_alert,
            "report": report_output.json_data if report_output else None,
        }

    def _generate_mock_segments(self, duration_sec: float) -> list[SegmentAnnotation]:
        """Generate mock annotated segments for MVP."""
        # Simple mock: divide audio into 2-3 segments
        num_segments = min(3, max(1, int(duration_sec / 10)))

        segments = []
        segment_duration_ms = int((duration_sec * 1000) / num_segments)

        labels = ["imprecise_consonants", "irregular_rate", "monopitch", "hypernasality"]

        for i in range(num_segments):
            start_ms = i * segment_duration_ms
            end_ms = (i + 1) * segment_duration_ms
            label = labels[i % len(labels)]
            weight = np.random.uniform(0.6, 0.9)

            segments.append(
                SegmentAnnotation(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    label=label,
                    weight=weight,
                )
            )

        return segments
