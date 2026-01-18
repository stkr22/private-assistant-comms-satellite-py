"""Basic tests for silero_vad module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from private_assistant_comms_satellite.silero_vad import SileroVad


class TestSileroVad:
    """Basic tests for SileroVad class."""

    @patch("private_assistant_comms_satellite.silero_vad.SileroVoiceActivityDetector")
    def test_silero_vad_initialization(self, mock_detector_class):
        """Test SileroVad initializes correctly."""
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        vad = SileroVad(threshold=0.5, trigger_level=2)

        assert vad.threshold == 0.5  # noqa: PLR2004
        assert vad.trigger_level == 2  # noqa: PLR2004
        assert vad._activation == 0
        assert vad.detector == mock_detector

    @patch("private_assistant_comms_satellite.silero_vad.SileroVoiceActivityDetector")
    def test_call_with_none_resets(self, mock_detector_class):
        """Test calling with None resets the detector."""
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        vad = SileroVad(threshold=0.5, trigger_level=2)
        vad._activation = 5  # Set some activation

        result = vad(None)

        assert result is False
        assert vad._activation == 0
        mock_detector.reset.assert_called_once()

    @patch("private_assistant_comms_satellite.silero_vad.SileroVoiceActivityDetector")
    def test_call_with_insufficient_audio_data(self, mock_detector_class):
        """Test calling with insufficient audio data raises ValueError."""
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        vad = SileroVad(threshold=0.5, trigger_level=2)

        # Provide less than required samples (need 512)
        short_audio = np.random.uniform(-1, 1, 100).astype(np.float32)  # Only 100 samples
        with pytest.raises(ValueError, match="Audio array must be at least 512 samples"):
            vad(short_audio)

    @patch("private_assistant_comms_satellite.silero_vad.SileroVoiceActivityDetector")
    def test_call_speech_detection_below_threshold(self, mock_detector_class):
        """Test speech detection below threshold."""
        mock_detector = MagicMock()
        mock_detector.process_samples.return_value = 0.3  # Below threshold of 0.5
        mock_detector_class.return_value = mock_detector

        vad = SileroVad(threshold=0.5, trigger_level=2)

        # Provide sufficient audio data (512 samples minimum)
        audio_data = np.random.uniform(-1, 1, 1024).astype(np.float32)
        result = vad(audio_data)

        assert result is False
        assert vad._activation == 0  # Should decrease/stay at 0

    @patch("private_assistant_comms_satellite.silero_vad.SileroVoiceActivityDetector")
    def test_call_speech_detection_above_threshold_not_triggered(self, mock_detector_class):
        """Test speech detection above threshold but not enough to trigger."""
        mock_detector = MagicMock()
        mock_detector.process_samples.return_value = 0.7  # Above threshold of 0.5
        mock_detector_class.return_value = mock_detector

        vad = SileroVad(threshold=0.5, trigger_level=3)

        audio_data = np.random.uniform(-1, 1, 1024).astype(np.float32)
        result = vad(audio_data)

        assert result is False
        assert vad._activation == 1  # Should increment but not trigger yet

    @patch("private_assistant_comms_satellite.silero_vad.SileroVoiceActivityDetector")
    def test_call_speech_detection_triggers(self, mock_detector_class):
        """Test speech detection that triggers activation."""
        mock_detector = MagicMock()
        mock_detector.process_samples.return_value = 0.8  # Above threshold
        mock_detector_class.return_value = mock_detector

        vad = SileroVad(threshold=0.5, trigger_level=2)
        vad._activation = 1  # Already at activation level 1

        audio_data = np.random.uniform(-1, 1, 1024).astype(np.float32)
        result = vad(audio_data)

        assert result is True
        assert vad._activation == 0  # Should reset after trigger

    @patch("private_assistant_comms_satellite.silero_vad.SileroVoiceActivityDetector")
    def test_call_multiple_chunks_max_probability(self, mock_detector_class):
        """Test that multiple chunks use maximum probability."""
        mock_detector = MagicMock()
        # Return different probabilities for different chunks
        mock_detector.process_samples.side_effect = [0.3, 0.8, 0.4]  # Max is 0.8
        mock_detector_class.return_value = mock_detector

        vad = SileroVad(threshold=0.5, trigger_level=1)

        # 1536 samples = 3 chunks of 512 samples each
        audio_data = np.random.uniform(-1, 1, 1536).astype(np.float32)
        result = vad(audio_data)

        assert result is True  # Should trigger because max(0.3, 0.8) = 0.8 > 0.5
        assert mock_detector.process_samples.call_count == 2  # noqa: PLR2004 - Should process 2 chunks (early termination)
