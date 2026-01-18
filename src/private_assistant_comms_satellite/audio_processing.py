"""Audio signal processing utilities for noise reduction and enhancement."""

import numpy as np
from scipy import signal


# AIDEV-NOTE: Parametric EQ using biquad filters for speech intelligibility enhancement
class ParametricEQ:
    """Parametric equalizer for voice enhancement.

    Boosts presence frequencies (3-4 kHz) to improve speech intelligibility and clarity.
    Optimized for wake word detection systems.
    """

    def __init__(
        self,
        sample_rate: int,
        presence_boost_db: float = 2.5,
        presence_freq_hz: float = 3500.0,
        presence_q: float = 2.5,
    ):
        """Initialize parametric EQ.

        Args:
            sample_rate: Audio sample rate in Hz
            presence_boost_db: Boost at presence frequency in dB (0-6)
            presence_freq_hz: Center frequency for presence (2000-5000 Hz)
            presence_q: Q factor for presence (0.5-5.0, higher = narrower)
        """
        self.sample_rate = sample_rate
        self.filters: list[np.ndarray] = []
        self.filter_states: list[np.ndarray | None] = []

        # Presence boost (peaking EQ at 3-4 kHz)
        if presence_boost_db > 0:
            # Get filter coefficients as (b, a) tuple
            b, a = signal.iirpeak(presence_freq_hz, presence_q, fs=sample_rate)
            # Convert to SOS format for numerical stability
            sos_presence = signal.tf2sos(b, a)
            # Apply gain scaling for dB boost
            gain_linear = 10 ** (presence_boost_db / 20)
            sos_presence = self._scale_filter_gain(sos_presence, gain_linear)
            self.filters.append(sos_presence)
            self.filter_states.append(None)

    def _scale_filter_gain(self, sos: np.ndarray, gain: float) -> np.ndarray:
        """Scale filter b coefficients by gain factor."""
        sos_scaled = sos.copy()
        sos_scaled[:, :3] *= gain  # Scale b0, b1, b2
        return sos_scaled  # type: ignore[no-any-return]

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply parametric EQ to audio chunk.

        Args:
            audio_chunk: Audio data as 1D numpy array (float32)

        Returns:
            EQ'd audio data
        """
        output = audio_chunk.copy()

        for i, sos in enumerate(self.filters):
            if self.filter_states[i] is None:
                # Initialize filter state
                self.filter_states[i] = signal.sosfilt_zi(sos) * output[0]

            # Apply filter with state preservation
            output, self.filter_states[i] = signal.sosfilt(sos, output, zi=self.filter_states[i])

        return output.astype(np.float32)  # type: ignore[no-any-return]

    def reset(self):
        """Reset filter states."""
        self.filter_states = [None] * len(self.filters)


# AIDEV-NOTE: Simple RMS-based AGC for volume normalization
class SimpleAGC:
    """Simple automatic gain control using RMS normalization.

    Normalizes audio volume to target RMS level with smoothed gain changes
    to avoid "pumping" artifacts.
    """

    def __init__(
        self,
        target_rms: float = 0.1,
        smoothing: float = 0.9,
        min_gain: float = 0.1,
        max_gain: float = 10.0,
    ):
        """Initialize AGC.

        Args:
            target_rms: Target RMS level (0.01-0.5)
            smoothing: Gain smoothing factor (0-0.99, higher = smoother)
            min_gain: Minimum gain factor (prevent over-attenuation)
            max_gain: Maximum gain factor (prevent over-amplification)
        """
        self.target_rms = target_rms
        self.smoothing = smoothing
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.current_gain = 1.0

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply AGC to audio chunk.

        Args:
            audio_chunk: Audio data as 1D numpy array (float32)

        Returns:
            Gain-adjusted audio data
        """
        # Calculate RMS of current chunk
        rms = np.sqrt(np.mean(audio_chunk**2))

        if rms > 1e-6:  # noqa: PLR2004  # Avoid division by near-zero
            # Calculate desired gain
            desired_gain = self.target_rms / rms

            # Clamp gain to safe range
            desired_gain = np.clip(desired_gain, self.min_gain, self.max_gain)

            # Smooth gain changes (exponential moving average)
            self.current_gain = self.smoothing * self.current_gain + (1 - self.smoothing) * desired_gain

        # Apply gain
        return (audio_chunk * self.current_gain).astype(np.float32)  # type: ignore[no-any-return]

    def reset(self):
        """Reset AGC state."""
        self.current_gain = 1.0
