import logging
import socket
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class Config(BaseModel):
    wakework_detection_threshold: float = 0.6
    openwakeword_inference_framework: str = "onnx"
    path_or_name_wakeword_model: str = "./hey_nova.onnx"
    name_wakeword_model: str = "hey_nova"
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
    chunk_size_ow: int = 1280
    vad_threshold: float = 0.6
    """Silence threshold (0-1, 1 is speech)"""
    vad_trigger: int = 1
    """Number of chunks to cross threshold before activation."""
    start_listening_path: str | None = None
    stop_listening_path: str | None = None


def load_config(config_path: Path) -> Config:
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
