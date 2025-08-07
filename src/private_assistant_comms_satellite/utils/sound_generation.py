"""Sound generation utilities for recording notifications.

This module provides functions to generate audio signals programmatically,
allowing the system to create notification sounds without requiring external WAV files.
All audio is generated as float32 arrays in the range [-1.0, 1.0] for sounddevice compatibility.
"""

import logging
import math

import numpy as np
import numpy.typing as np_typing

logger = logging.getLogger(__name__)


def generate_sweep(
    start_freq: float,
    end_freq: float,
    duration: float,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> np_typing.NDArray[np.float32]:
    """Generate a frequency sweep (chirp) from start to end frequency.

    Args:
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        duration: Sweep duration in seconds
        sample_rate: Audio sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        Float32 audio array with frequency sweep
    """
    # AIDEV-NOTE: Linear frequency sweep with smooth transitions
    num_samples = int(sample_rate * duration)
    fade_duration = 0.02  # Fixed fade duration
    fade_samples = int(sample_rate * fade_duration)

    t = np.linspace(0, duration, num_samples, False)

    # Linear frequency interpolation
    freq_t = start_freq + (end_freq - start_freq) * (t / duration)

    # Generate wave with varying frequency
    # Use cumulative sum of frequency to get proper phase continuity
    phase = 2 * math.pi * np.cumsum(freq_t) / sample_rate
    wave = amplitude * np.sin(phase)

    # Apply fade-in
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        wave[:fade_samples] *= fade_in

    # Apply fade-out
    if fade_samples > 0:
        fade_out = np.linspace(1, 0, fade_samples)
        wave[-fade_samples:] *= fade_out

    # Keep as float32 for sounddevice
    wave_float32: np_typing.NDArray[np.float32] = wave.astype(np.float32)

    logger.debug(
        "Generated sweep: %.1f-%.1f Hz, %.2f seconds, %d samples", start_freq, end_freq, duration, len(wave_float32)
    )
    return wave_float32


def generate_start_recording_sound(sample_rate: int = 16000) -> np_typing.NDArray[np.float32]:
    """Generate a pleasant start recording notification sound using upward frequency sweep.

    Args:
        sample_rate: Audio sample rate in Hz

    Returns:
        Float32 audio array suitable for sounddevice playback
    """
    # AIDEV-NOTE: Optimistic upward sweep to signal recording start
    wave = generate_sweep(200, 800, 0.4, sample_rate, amplitude=0.4)
    logger.info("Generated start recording sound: sweep_up style, %d samples", len(wave))
    return wave


def generate_stop_recording_sound(sample_rate: int = 16000) -> np_typing.NDArray[np.float32]:
    """Generate a pleasant stop recording notification sound using downward frequency sweep.

    Args:
        sample_rate: Audio sample rate in Hz

    Returns:
        Float32 audio array suitable for sounddevice playback
    """
    # AIDEV-NOTE: Gentle downward sweep to signal recording completion
    wave = generate_sweep(800, 200, 0.4, sample_rate, amplitude=0.4)
    logger.info("Generated stop recording sound: sweep_down style, %d samples", len(wave))
    return wave


def generate_default_sounds(
    sample_rate: int = 16000,
) -> tuple[np_typing.NDArray[np.float32], np_typing.NDArray[np.float32]]:
    """Generate both start and stop recording sounds using frequency sweeps.

    Args:
        sample_rate: Audio sample rate in Hz

    Returns:
        Tuple of (start_sound_array, stop_sound_array) as float32 arrays
    """
    # AIDEV-NOTE: Convenience function for generating paired sweep notification sounds
    start_sound = generate_start_recording_sound(sample_rate)
    stop_sound = generate_stop_recording_sound(sample_rate)

    logger.info("Generated default sound pair: start=sweep_up, stop=sweep_down")
    return start_sound, stop_sound
