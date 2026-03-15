"""Speech analysis endpoint."""

import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional

from api.schemas import AnalyseResponse, ErrorResponse
from api.dependencies import get_pipeline
from src.pipeline import SlurringDetectionPipeline
from src.scoring.schemas import SeverityLevel
from src.risk.schemas import RiskTier
from src.reporting.schemas import SegmentAnnotation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/speech", tags=["analysis"])


@router.post("/analyse", response_model=AnalyseResponse)
async def analyse_speech(
    audio_file: UploadFile = File(..., description="Audio file (wav, mp3, ogg, m4a)"),
    patient_age: Optional[int] = Form(None, ge=18, le=120),
    onset_hours: Optional[float] = Form(None, ge=0, le=168),
    return_pdf: bool = Form(True),
    pipeline: SlurringDetectionPipeline = Depends(get_pipeline),
):
    """
    Analyze speech for dysarthria and stroke risk.

    Args:
        audio_file: Audio file upload
        patient_age: Patient age (optional, improves risk scoring)
        onset_hours: Hours since symptom onset (optional)
        return_pdf: Generate PDF report (not implemented in MVP)
        pipeline: Pipeline dependency

    Returns:
        AnalyseResponse with results
    """
    logger.info(f"Received analysis request: {audio_file.filename}")

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
            content = await audio_file.read()
            logger.info(f"Received {len(content)} bytes, content type: {type(content)}")
            logger.info(f"First 20 bytes: {content[:20].hex() if len(content) >= 20 else content.hex()}")

            bytes_written = tmp_file.write(content)
            tmp_file.flush()  # Ensure data is written to disk
            tmp_file_path = tmp_file.name

            logger.info(f"Wrote {bytes_written} bytes to {tmp_file_path}")

            # Verify file was written correctly
            file_size = Path(tmp_file_path).stat().st_size
            logger.info(f"Verified file size on disk: {file_size} bytes")

        try:
            # Run pipeline
            result = pipeline.analyse(
                audio_file=tmp_file_path,
                patient_age=patient_age,
                onset_hours=onset_hours,
                return_report=True,
            )

            # Convert segments
            segments = [
                SegmentAnnotation(**seg) for seg in result["segments"]
            ]

            # Build response
            response = AnalyseResponse(
                request_id=result["request_id"],
                slurring_score=result["slurring_score"],
                severity=SeverityLevel(result["severity"]),
                risk_score=result["risk_score"],
                risk_tier=RiskTier(result["risk_tier"]),
                confidence=result["confidence"],
                segments=segments,
                acoustic_summary=result["acoustic_summary"],
                json_report=result["report"],
                processing_time_ms=result["processing_time_ms"],
                model_version=result["model_version"],
                emergency_alert=result["emergency_alert"],
            )

            logger.info(
                f"Analysis complete: {result['request_id']} "
                f"(score={result['slurring_score']}, tier={result['risk_tier']})"
            )

            return response

        finally:
            # Clean up temp file
            Path(tmp_file_path).unlink(missing_ok=True)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
