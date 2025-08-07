import asyncio
import os
import pathlib
import wave

import numpy as np
import openwakeword
import openwakeword.utils
from private_assistant_commons import messages, skill_logger

from private_assistant_comms_satellite import satellite
from private_assistant_comms_satellite.utils import (
    config,
    mqtt_utils,
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


def ensure_openwakeword_models() -> None:
    """Ensure OpenWakeWord models are downloaded for first-time setup.

    Downloads spectrograms and silero VAD model if not already present.
    This prevents the need for manual model download before first use.
    """
    logger = skill_logger.SkillLogger.get_logger("Private Assistant Comms Satellite")

    logger.info("Ensuring OpenWakeWord base models are available...")
    try:
        # Download base models (spectrograms and silero VAD) - only downloads if missing
        openwakeword.utils.download_models(model_names=["none"])
        logger.debug("OpenWakeWord base models verified successfully")
    except Exception as e:
        logger.error("Failed to download OpenWakeWord models: %s", e)
        logger.error("Please download manually")
        raise


def start_satellite(config_path: pathlib.Path) -> None:
    """Start the satellite application with the given configuration.

    Args:
        config_path: Path to the YAML configuration file
    """
    # AIDEV-NOTE: Refactored from main() to allow CLI integration
    # Ensure base models are available before loading config and creating wakeword model
    ensure_openwakeword_models()

    config_obj = config.load_config(config_path)
    wakeword_model = openwakeword.Model(
        wakeword_models=[config_obj.path_or_name_wakeword_model],
        enable_speex_noise_suppression=True,
        vad_threshold=config_obj.vad_threshold,
        inference_framework=config_obj.openwakeword_inference_framework,
    )
    topic_to_queue: dict[str, asyncio.Queue[messages.Response]] = {}
    output_queue: asyncio.Queue[messages.Response] = asyncio.Queue()
    output_topic = config_obj.output_topic
    topic_to_queue[output_topic] = output_queue
    topic_to_queue[config_obj.broadcast_topic] = output_queue

    mqtt_client = mqtt_utils.AsyncMQTTClient(
        hostname=config_obj.mqtt_server_host,
        port=config_obj.mqtt_server_port,
        client_id=config_obj.client_id,
        topic_to_queue=topic_to_queue,
        use_websockets=config_obj.mqtt_use_websockets,
        websocket_path=config_obj.mqtt_websocket_path,
        use_ssl=config_obj.mqtt_use_ssl,
    )

    logger = skill_logger.SkillLogger.get_logger("Private Assistant Comms Satellite")
    # Preload sounds with fallback to generated sounds
    start_listening_sound = load_sound_with_fallback(config_obj.start_listening_path, "start", config_obj.samplerate)
    stop_listening_sound = load_sound_with_fallback(config_obj.stop_listening_path, "stop", config_obj.samplerate)
    satellite_handler = satellite.Satellite(
        config=config_obj,
        output_queue=output_queue,
        start_listening_sound=start_listening_sound,
        stop_listening_sound=stop_listening_sound,
        wakeword_model=wakeword_model,
        mqtt_client=mqtt_client,
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
