"""Audio file loading and validation."""

import logging
import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

from src.ingestion.schemas import AudioInput

logger = logging.getLogger(__name__)


class AudioLoader:
    """Load and validate audio files in various formats."""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".webm"}
    MIME_TYPE_MAP = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/m4a",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }

    def __init__(self, max_duration: float = 60.0, min_duration: float = 5.0):
        """
        Initialize audio loader.

        Args:
            max_duration: Maximum allowed audio duration (seconds)
            min_duration: Minimum allowed audio duration (seconds)
        """
        self.max_duration = max_duration
        self.min_duration = min_duration

    def load(self, file_path: str | Path) -> Tuple[AudioInput, np.ndarray]:
        """
        Load audio file and return metadata + waveform.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (AudioInput metadata, waveform array)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format unsupported or duration invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Validate file format
        file_ext = file_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        logger.info(f"Loading audio file: {file_path}")

        try:
            # Formats that need conversion to WAV via pydub/ffmpeg
            needs_conversion = {".webm", ".ogg", ".m4a", ".mp3"}

            if file_ext in needs_conversion:
                # Detect actual format from magic bytes (file extension can be misleading)
                with open(file_path, 'rb') as f:
                    magic_bytes = f.read(12)

                # EBML header (0x1a45dfa3) = WebM/Matroska
                # OGG header (0x4f676753 = "OggS")
                # MP3 frame sync (0xfffb / 0xfff3) or ID3 tag (0x494433)
                # M4A/MP4 — ftyp box at offset 4 (bytes 4-7 = "ftyp")
                if magic_bytes[:4] == b'\x1a\x45\xdf\xa3':
                    format_name = "webm"
                elif magic_bytes[:4] == b'OggS':
                    format_name = "ogg"
                elif magic_bytes[4:8] == b'ftyp':
                    format_name = "m4a"
                elif magic_bytes[:3] == b'ID3' or magic_bytes[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
                    format_name = "mp3"
                else:
                    # Fallback to extension-based detection
                    format_name = file_ext.lstrip(".")
                    logger.warning(f"Unknown magic bytes {magic_bytes[:4].hex()}, assuming {format_name.upper()}")

                logger.info(f"Detected {format_name.upper()} format, converting to WAV...")
                audio = AudioSegment.from_file(str(file_path), format=format_name)

                # Create temporary WAV file
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio.export(temp_wav.name, format="wav")
                temp_wav.close()

                # Use the converted file for processing
                processing_path = Path(temp_wav.name)
                logger.info(f"{format_name.upper()} converted to temporary WAV: {processing_path}")
            else:
                processing_path = file_path

            # Load audio using librosa
            waveform, sample_rate = librosa.load(processing_path, sr=None, mono=False)

            # Get audio info using soundfile for more accurate metadata
            info = sf.info(processing_path)
            duration_sec = info.duration
            num_channels = info.channels

            # Clean up temporary file if we created one
            if file_ext in needs_conversion:
                try:
                    processing_path.unlink()
                    logger.debug("Cleaned up temporary WAV file")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

            # Validate duration
            if duration_sec < self.min_duration:
                raise ValueError(
                    f"Audio duration ({duration_sec:.1f}s) is below minimum "
                    f"({self.min_duration}s)"
                )

            if duration_sec > self.max_duration:
                raise ValueError(
                    f"Audio duration ({duration_sec:.1f}s) exceeds maximum "
                    f"({self.max_duration}s)"
                )

            # If stereo, take mean to get mono (librosa will do this if mono=True)
            if waveform.ndim == 2:
                logger.debug(f"Converting stereo to mono (channels: {num_channels})")
                waveform = np.mean(waveform, axis=0)

            # Create AudioInput metadata
            audio_input = AudioInput(
                file_path=file_path,
                file_name=file_path.name,
                file_size_bytes=file_path.stat().st_size,
                mime_type=self.MIME_TYPE_MAP[file_ext],
                sample_rate=int(sample_rate),
                duration_sec=duration_sec,
                num_channels=num_channels,
            )

            logger.info(
                f"Loaded audio: {duration_sec:.2f}s, {sample_rate}Hz, "
                f"{num_channels} channel(s), {waveform.shape[0]} samples"
            )

            return audio_input, waveform

        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            raise ValueError(f"Error loading audio file: {e}") from e

    def load_from_bytes(
        self, audio_bytes: bytes, file_name: str = "upload.wav", sr: int = None
    ) -> Tuple[AudioInput, np.ndarray]:
        """
        Load audio from byte stream (for uploads).

        Args:
            audio_bytes: Audio data as bytes
            file_name: Original filename (for format detection)
            sr: Target sample rate (if None, use original)

        Returns:
            Tuple of (AudioInput metadata, waveform array)
        """
        import io

        logger.info(f"Loading audio from bytes: {file_name}")

        # Detect format from filename
        file_ext = Path(file_name).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            # Load from bytes using soundfile
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = sf.read(audio_buffer, dtype="float32")

            # Get duration
            duration_sec = len(waveform) / sample_rate

            # Validate duration
            if duration_sec < self.min_duration or duration_sec > self.max_duration:
                raise ValueError(
                    f"Audio duration ({duration_sec:.1f}s) outside valid range "
                    f"({self.min_duration}s - {self.max_duration}s)"
                )

            # Convert to mono if stereo
            if waveform.ndim == 2:
                waveform = np.mean(waveform, axis=1)
                num_channels = 2
            else:
                num_channels = 1

            # Resample if requested
            if sr is not None and sr != sample_rate:
                logger.debug(f"Resampling from {sample_rate}Hz to {sr}Hz")
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=sr)
                sample_rate = sr

            # Create AudioInput metadata
            audio_input = AudioInput(
                file_path=None,
                file_name=file_name,
                file_size_bytes=len(audio_bytes),
                mime_type=self.MIME_TYPE_MAP[file_ext],
                sample_rate=int(sample_rate),
                duration_sec=duration_sec,
                num_channels=num_channels,
            )

            logger.info(f"Loaded audio from bytes: {duration_sec:.2f}s, {sample_rate}Hz")

            return audio_input, waveform

        except Exception as e:
            logger.error(f"Failed to load audio from bytes: {e}")
            raise ValueError(f"Error loading audio from bytes: {e}") from e

    @staticmethod
    def get_audio_info(file_path: str | Path) -> dict:
        """
        Get audio file information without loading waveform.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            info = sf.info(file_path)

            return {
                "file_name": file_path.name,
                "file_size_bytes": file_path.stat().st_size,
                "sample_rate": info.samplerate,
                "duration_sec": info.duration,
                "num_channels": info.channels,
                "format": info.format,
                "subtype": info.subtype,
            }

        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            raise ValueError(f"Error reading audio file: {e}") from e
