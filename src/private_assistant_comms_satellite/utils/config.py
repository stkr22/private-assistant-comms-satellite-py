"""Configuration models and YAML loading."""

import logging
import socket
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class AudioProcessingConfig(BaseModel):
    """Audio enhancement configuration."""

    # Parametric EQ for voice enhancement
    enable_voice_eq: bool = Field(
        default=True,
        description="Enable parametric EQ for voice enhancement",
    )
    eq_presence_boost_db: float = Field(
        default=3.0,
        ge=0.0,
        le=6.0,
        description="Presence boost at 3-4 kHz in dB (improves clarity)",
    )
    eq_presence_freq_hz: float = Field(
        default=3500.0,
        ge=2000.0,
        le=5000.0,
        description="Center frequency for presence boost",
    )
    eq_presence_q: float = Field(
        default=2.5,
        ge=0.5,
        le=5.0,
        description="Q factor for presence boost (higher = narrower)",
    )

    # Automatic Gain Control
    enable_agc: bool = Field(
        default=True,
        description="Enable automatic gain control",
    )
    agc_target_rms: float = Field(
        default=0.16,
        ge=0.01,
        le=0.5,
        description="Target RMS level for AGC normalization",
    )
    agc_smoothing: float = Field(
        default=0.7,
        ge=0.0,
        le=0.99,
        description="Gain smoothing factor (higher = smoother, slower)",
    )


class Config(BaseModel):
    """Satellite application configuration."""

    wakework_detection_threshold: float = 0.6
    wakeword_model_path: str = "./okay_nabu.tflite"
    wakeword_sliding_window_size: int = Field(default=5, ge=1, le=20)
    wakeword_cooldown_chunks: int = Field(default=40, ge=1, le=200)
    # AIDEV-NOTE: Ground station WebSocket configuration replaces MQTT and API endpoints
    ground_station_url: str = "ws://localhost:8000/satellite"
    client_id: str = socket.gethostname()
    room: str = "livingroom"
    output_device_index: int = 1
    input_device_index: int = 1
    max_command_input_seconds: int = 15
    max_length_speech_pause: float = 1.0
    samplerate: int = 16000
    chunk_size: int = 512
    vad_threshold: float = 0.6
    """Silence threshold (0-1, 1 is speech)"""
    vad_trigger: int = 1
    """Number of chunks to cross threshold before activation."""
    start_listening_path: str | None = None
    stop_listening_path: str | None = None
    audio_processing: AudioProcessingConfig = Field(
        default_factory=AudioProcessingConfig,
        description="Audio signal processing configuration",
    )


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from a YAML file."""
    try:
        with config_path.open("r") as file:
            config_data = yaml.safe_load(file)
        return Config.model_validate(config_data)
    except FileNotFoundError as err:
        logger.error("Config file not found: %s", config_path)
        raise err
    except ValidationError as err_v:
        logger.error("Validation error: %s", err_v)
        raise err_v
