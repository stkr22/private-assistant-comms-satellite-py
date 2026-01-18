"""Debug utilities for recording and analyzing audio quality."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records audio samples for debugging and quality analysis."""

    def __init__(self, output_dir: str = "/tmp/audio_debug"):
        """Initialize audio recorder.

        Args:
            output_dir: Directory to save audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Audio debug recorder initialized: %s", self.output_dir)

    def save_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        prefix: str = "audio",
    ) -> Path:
        """Save audio data to WAV file.

        Args:
            audio_data: Audio samples (float32 or int16)
            sample_rate: Sample rate in Hz
            prefix: Filename prefix

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.wav"
        filepath = self.output_dir / filename

        # Ensure float32 for soundfile
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        sf.write(filepath, audio_data, sample_rate)
        logger.info(
            "Saved audio: %s (%s samples, %.2fs)",
            filepath,
            len(audio_data),
            len(audio_data) / sample_rate,
        )

        return filepath

    def save_comparison(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        sample_rate: int,
        label: str = "comparison",
    ):
        """Save before/after comparison.

        Args:
            original: Original audio samples
            processed: Processed audio samples
            sample_rate: Sample rate in Hz
            label: Label for the comparison
        """
        self.save_audio(original, sample_rate, f"{label}_original")
        self.save_audio(processed, sample_rate, f"{label}_processed")
        logger.info("Saved comparison: %s", label)


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB.

    Args:
        signal: Signal samples
        noise: Noise samples

    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return float("inf")

    return float(10 * np.log10(signal_power / noise_power))
