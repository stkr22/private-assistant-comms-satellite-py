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
    speech_transcription_api: str = "http://localhost:8000/transcribe"
    speech_transcription_api_token: str | None = None
    speech_synthesis_api: str = "http://localhost:8080/synthesizeSpeech"
    speech_synthesis_api_token: str | None = None
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
    mqtt_server_host: str = "localhost"
    mqtt_server_port: int = 1883
    mqtt_use_websockets: bool = False
    mqtt_websocket_path: str = "/mqtt"
    mqtt_use_ssl: bool = False
    broadcast_topic: str = "assistant/comms_bridge/broadcast"
    base_topic_overwrite: str | None = None
    input_topic_overwrite: str | None = None
    output_topic_overwrite: str | None = None
    start_listening_path: str | None = None
    stop_listening_path: str | None = None

    @property
    def base_topic(self) -> str:
        """Generate base MQTT topic for this satellite instance."""
        # AIDEV-NOTE: Dynamic topic generation based on client_id for multi-satellite deployments
        return self.base_topic_overwrite or f"assistant/comms_bridge/all/{self.client_id}"

    @property
    def input_topic(self) -> str:
        """Topic where satellite publishes recognized speech."""
        return self.input_topic_overwrite or f"{self.base_topic}/input"

    @property
    def output_topic(self) -> str:
        """Topic where satellite subscribes for TTS responses."""
        return self.output_topic_overwrite or f"{self.base_topic}/output"


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
