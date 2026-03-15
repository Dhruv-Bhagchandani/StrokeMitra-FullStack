"""PDF report renderer (placeholder for MVP)."""

import logging
from pathlib import Path

from src.reporting.schemas import ReportData, ReportOutput

logger = logging.getLogger(__name__)


class PDFRenderer:
    """Render report as PDF (placeholder for MVP)."""

    def render(self, report_data: ReportData, save_path: Path | None = None) -> ReportOutput:
        """
        Render report as PDF (placeholder).

        Args:
            report_data: Report data
            save_path: Optional path to save PDF file

        Returns:
            ReportOutput (PDF not implemented in MVP)
        """
        logger.warning("PDF rendering not implemented in MVP. Use JSON renderer instead.")

        # Return empty PDF output
        return ReportOutput(
            report_id=report_data.report_id,
            pdf_bytes=None,
            pdf_path=None,
        )
