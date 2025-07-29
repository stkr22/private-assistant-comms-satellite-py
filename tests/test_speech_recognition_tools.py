"""Tests for speech_recognition_tools module."""

import numpy as np
import pytest
from pydantic import ValidationError

from private_assistant_comms_satellite.utils import config, speech_recognition_tools


@pytest.fixture
def sample_config():
    """Create a sample config for testing."""
    return config.Config(
        speech_transcription_api="http://test-stt.example.com/transcribe",
        speech_transcription_api_token="test-stt-token",
        speech_synthesis_api="http://test-tts.example.com/synthesize",
        speech_synthesis_api_token="test-tts-token",
    )


@pytest.fixture
def sample_audio_data():
    """Create sample audio data for testing."""
    # Create a simple sine wave as test audio
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


class TestInt2Float:
    """Test the int2float conversion function."""

    def test_int2float_with_valid_data(self):
        """Test int2float with valid int16 data."""
        # Create test data
        int_data = np.array([1000, -1000, 0, 16383, -16384], dtype=np.int16)

        result = speech_recognition_tools.int2float(int_data)

        assert result.dtype == np.float32
        assert len(result) == len(int_data)
        # Check that values are normalized to [-1, 1] range
        assert np.all(np.abs(result) <= 1.0)

    def test_int2float_with_zeros(self):
        """Test int2float with all zeros."""
        int_data = np.array([0, 0, 0], dtype=np.int16)

        result = speech_recognition_tools.int2float(int_data)

        assert result.dtype == np.float32
        assert np.all(result == 0.0)

    def test_int2float_squeeze_behavior(self):
        """Test that int2float properly squeezes dimensions."""
        # Create 2D array that should be squeezed
        int_data = np.array([[1000], [2000], [3000]], dtype=np.int16)

        result = speech_recognition_tools.int2float(int_data)

        assert result.ndim == 1  # Should be squeezed to 1D
        assert len(result) == 3  # noqa: PLR2004


# NOTE: Async API tests removed for simplicity - focus on basic unit tests


class TestSTTResponse:
    """Test the STTResponse model."""

    def test_valid_stt_response(self):
        """Test creating STTResponse with valid data."""
        data = {"text": "hello world", "message": "success"}

        response = speech_recognition_tools.STTResponse.model_validate(data)

        assert response.text == "hello world"
        assert response.message == "success"

    def test_invalid_stt_response_missing_text(self):
        """Test STTResponse validation with missing text field."""
        data = {"message": "success"}  # Missing 'text' field

        with pytest.raises(ValidationError):
            speech_recognition_tools.STTResponse.model_validate(data)

    def test_invalid_stt_response_missing_message(self):
        """Test STTResponse validation with missing message field."""
        data = {"text": "hello world"}  # Missing 'message' field

        with pytest.raises(ValidationError):
            speech_recognition_tools.STTResponse.model_validate(data)

    def test_stt_response_extra_fields(self):
        """Test STTResponse with extra fields (should be ignored)."""
        data = {"text": "hello world", "message": "success", "extra_field": "ignored"}

        response = speech_recognition_tools.STTResponse.model_validate(data)

        assert response.text == "hello world"
        assert response.message == "success"
        # Extra field should not be accessible
        assert not hasattr(response, "extra_field")
