"""Application entry point for the satellite."""

import os
import pathlib
import wave

import numpy as np
from private_assistant_commons import skill_logger

from private_assistant_comms_satellite import satellite
from private_assistant_comms_satellite.micro_wake_word import MicroWakeWord
from private_assistant_comms_satellite.utils import (
    config,
    sound_generation,
)


# Load WAV file into memory
def load_wav_file(file_path: str) -> bytes:
    """Load WAV file into memory using context manager."""
    with wave.open(file_path, "rb") as wf:
        return wf.readframes(wf.getnframes())


def load_sound_with_fallback(
    file_path: str | None,
    sound_type: str,
    sample_rate: int = 16000,
) -> bytes | np.ndarray:
    """Load sound from file or generate if path is None.

    Args:
        file_path: Path to WAV file, or None to use generated sound
        sound_type: Either 'start' or 'stop' for the appropriate generated sound
        sample_rate: Sample rate for generated sounds

    Returns:
        Audio bytes (from file) or float32 numpy array (generated) suitable for sounddevice

    """
    logger = skill_logger.SkillLogger.get_logger("Private Assistant Comms Satellite")

    # If path is None, use generated sounds
    if file_path is None:
        logger.debug("No file path configured, using generated %s sound", sound_type)
        if sound_type == "start":
            return sound_generation.generate_start_recording_sound(sample_rate=sample_rate)
        if sound_type == "stop":
            return sound_generation.generate_stop_recording_sound(sample_rate=sample_rate)
        raise ValueError(f"Unknown sound_type: {sound_type}")

    # If path is provided, load from file
    if pathlib.Path(file_path).exists():
        try:
            logger.debug("Loading %s sound from file: %s", sound_type, file_path)
            return load_wav_file(file_path)
        except Exception as e:
            logger.error("Failed to load %s sound from %s: %s", sound_type, file_path, e)
            raise
    else:
        logger.error("Sound file not found: %s", file_path)
        raise FileNotFoundError(f"Sound file not found: {file_path}")


def start_satellite(config_path: pathlib.Path) -> None:
    """Start the satellite application with the given configuration.

    Args:
        config_path: Path to the YAML configuration file

    """
    config_obj = config.load_config(config_path)

    model_path = pathlib.Path(config_obj.wakeword_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Wake word model not found: {model_path}. "
            "Download a .tflite model from https://github.com/esphome/micro-wake-word-models"
        )

    wakeword_model = MicroWakeWord(
        model_path=config_obj.wakeword_model_path,
        sliding_window_size=config_obj.wakeword_sliding_window_size,
        cooldown_chunks=config_obj.wakeword_cooldown_chunks,
        rearm_threshold=config_obj.wakework_detection_threshold,
    )

    logger = skill_logger.SkillLogger.get_logger("Private Assistant Comms Satellite")
    # Preload sounds with fallback to generated sounds
    start_listening_sound = load_sound_with_fallback(config_obj.start_listening_path, "start", config_obj.samplerate)
    stop_listening_sound = load_sound_with_fallback(config_obj.stop_listening_path, "stop", config_obj.samplerate)
    # Generate disconnection warning sound
    disconnection_sound = sound_generation.generate_disconnection_warning_sound(config_obj.samplerate)

    satellite_handler = satellite.Satellite(
        config=config_obj,
        start_listening_sound=start_listening_sound,
        stop_listening_sound=stop_listening_sound,
        disconnection_sound=disconnection_sound,
        wakeword_model=wakeword_model,
        logger=logger,
    )

    try:
        satellite_handler.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, stopping...")
    finally:
        satellite_handler.cleanup()


if __name__ == "__main__":
    # AIDEV-NOTE: Direct execution for development, use CLI for production
    config_path = pathlib.Path(os.getenv("PRIVATE_ASSISTANT_API_CONFIG_PATH", "local_config.yaml"))
    start_satellite(config_path)
