"""Basic tests for config module."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from private_assistant_comms_satellite.utils.config import Config, load_config


class TestConfig:
    """Basic tests for Config model."""

    def test_config_defaults(self):
        """Test that Config initializes with expected defaults."""
        config = Config()

        assert config.wakework_detection_threshold == 0.6  # noqa: PLR2004
        assert config.path_or_name_wakeword_model == "./hey_nova.onnx"
        assert config.name_wakeword_model == "hey_nova"
        assert config.samplerate == 16000  # noqa: PLR2004
        assert config.chunk_size == 512  # noqa: PLR2004
        assert config.vad_threshold == 0.6  # noqa: PLR2004
        assert config.mqtt_server_host == "localhost"
        assert config.mqtt_server_port == 1883  # noqa: PLR2004

    def test_config_with_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            wakework_detection_threshold=0.8,
            samplerate=22050,
            mqtt_server_host="custom.mqtt.server",
            mqtt_server_port=8883,
        )

        assert config.wakework_detection_threshold == 0.8  # noqa: PLR2004
        assert config.samplerate == 22050  # noqa: PLR2004
        assert config.mqtt_server_host == "custom.mqtt.server"
        assert config.mqtt_server_port == 8883  # noqa: PLR2004

    def test_config_topic_properties(self):
        """Test computed topic properties."""
        config = Config(client_id="test_client")

        expected_base = "assistant/comms_bridge/all/test_client"
        assert config.base_topic == expected_base
        assert config.input_topic == f"{expected_base}/input"
        assert config.output_topic == f"{expected_base}/output"

    def test_config_topic_overwrites(self):
        """Test topic overwrites work correctly."""
        config = Config(
            base_topic_overwrite="custom/base",
            input_topic_overwrite="custom/input",
            output_topic_overwrite="custom/output",
        )

        assert config.base_topic == "custom/base"
        assert config.input_topic == "custom/input"
        assert config.output_topic == "custom/output"


class TestLoadConfig:
    """Basic tests for config loading."""

    def test_load_valid_config(self):
        """Test loading a valid YAML config file."""
        config_data = """
wakework_detection_threshold: 0.7
samplerate: 22050
mqtt_server_host: "test.server"
mqtt_server_port: 8883
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            f.flush()

            config_path = Path(f.name)
            config = load_config(config_path)

            assert config.wakework_detection_threshold == 0.7  # noqa: PLR2004
            assert config.samplerate == 22050  # noqa: PLR2004
            assert config.mqtt_server_host == "test.server"
            assert config.mqtt_server_port == 8883  # noqa: PLR2004

            # Clean up
            config_path.unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file raises FileNotFoundError."""
        non_existent_path = Path("does_not_exist.yaml")

        with pytest.raises(FileNotFoundError):
            load_config(non_existent_path)

    def test_load_invalid_config(self):
        """Test loading invalid config raises ValidationError."""
        invalid_config = """
wakework_detection_threshold: "not_a_number"
samplerate: -1000
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_config)
            f.flush()

            config_path = Path(f.name)

            with pytest.raises(ValidationError):
                load_config(config_path)

            # Clean up
            config_path.unlink()

    def test_load_empty_config_uses_defaults(self):
        """Test loading empty config file uses defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{}")  # Empty YAML
            f.flush()

            config_path = Path(f.name)
            config = load_config(config_path)

            # Should use defaults
            assert config.wakework_detection_threshold == 0.6  # noqa: PLR2004
            assert config.samplerate == 16000  # noqa: PLR2004

            # Clean up
            config_path.unlink()
