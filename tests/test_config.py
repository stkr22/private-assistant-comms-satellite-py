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

        assert config.wakework_detection_threshold == 0.6
        assert config.wakeword_model_path == "./okay_nabu.tflite"
        assert config.wakeword_sliding_window_size == 5
        assert config.wakeword_cooldown_chunks == 40
        assert config.samplerate == 16000
        assert config.chunk_size == 512
        assert config.vad_threshold == 0.6
        assert config.ground_station_url == "ws://localhost:8000/satellite"

    def test_config_with_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            wakework_detection_threshold=0.8,
            samplerate=22050,
            ground_station_url="wss://custom.ground.station/satellite",
        )

        assert config.wakework_detection_threshold == 0.8
        assert config.samplerate == 22050
        assert config.ground_station_url == "wss://custom.ground.station/satellite"


class TestLoadConfig:
    """Basic tests for config loading."""

    def test_load_valid_config(self):
        """Test loading a valid YAML config file."""
        config_data = """
wakework_detection_threshold: 0.7
samplerate: 22050
ground_station_url: "wss://test.ground.station/satellite"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            f.flush()

            config_path = Path(f.name)
            config = load_config(config_path)

            assert config.wakework_detection_threshold == 0.7
            assert config.samplerate == 22050
            assert config.ground_station_url == "wss://test.ground.station/satellite"

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
            assert config.wakework_detection_threshold == 0.6
            assert config.samplerate == 16000

            # Clean up
            config_path.unlink()
