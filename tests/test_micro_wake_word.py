"""Tests for MicroWakeWord streaming wake word detector."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from private_assistant_comms_satellite.micro_wake_word import MicroWakeWord


def _make_mock_interpreter(input_feature_slices: int = 40, num_features: int = 40):
    """Create a mock TFLite interpreter with configurable input shape."""
    mock = MagicMock()
    mock.get_input_details.return_value = [
        {
            "shape": np.array([1, input_feature_slices, num_features]),
            "index": 0,
            "dtype": np.float32,
            "quantization": (0.0, 0),
        },
    ]
    mock.get_output_details.return_value = [
        {
            "shape": np.array([1, 1]),
            "index": 1,
            "dtype": np.float32,
            "quantization": (0.0, 0),
        },
    ]
    return mock


def _make_mock_frontend_result(features: list[float] | None = None, samples_read: int = 160):
    """Create a mock MicroFrontendOutput."""
    result = MagicMock()
    result.features = features if features is not None else []
    result.samples_read = samples_read
    return result


class TestMicroWakeWordInit:
    """Tests for MicroWakeWord initialization."""

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_initialization(self, mock_interpreter_cls, mock_frontend_cls):
        mock_interpreter_cls.return_value = _make_mock_interpreter(input_feature_slices=50, num_features=40)

        mww = MicroWakeWord(model_path="fake.tflite", sliding_window_size=7, cooldown_chunks=30)

        assert mww._input_feature_slices == 50
        assert mww._num_features == 40
        mock_interpreter_cls.return_value.allocate_tensors.assert_called_once()
        mock_frontend_cls.assert_called_once()


class TestMicroWakeWordPredict:
    """Tests for MicroWakeWord.predict()."""

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_insufficient_features_returns_zero(self, mock_interpreter_cls, mock_frontend_cls):
        """When not enough features have accumulated, predict returns 0.0."""
        mock_interpreter = _make_mock_interpreter(input_feature_slices=40)
        mock_interpreter_cls.return_value = mock_interpreter

        # Frontend returns one feature frame per call (not enough for 40 slices)
        mock_frontend = MagicMock()
        mock_frontend.process_samples.return_value = _make_mock_frontend_result(features=[0.1] * 40, samples_read=160)
        mock_frontend_cls.return_value = mock_frontend

        mww = MicroWakeWord(model_path="fake.tflite")

        # Feed one 512-sample chunk — produces ~3 feature frames, far less than 40
        audio = np.zeros(512, dtype=np.int16)
        prob = mww.predict(audio)

        assert prob == 0.0
        # Interpreter should NOT have been invoked (not enough features)
        mock_interpreter.invoke.assert_not_called()

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_returns_probability_when_buffer_full(self, mock_interpreter_cls, mock_frontend_cls):
        """When feature buffer is full, predict runs inference and returns smoothed probability."""
        mock_interpreter = _make_mock_interpreter(input_feature_slices=3)
        mock_interpreter_cls.return_value = mock_interpreter

        # Set up interpreter to return probability 0.8
        mock_interpreter.get_tensor.return_value = np.array([[0.8]], dtype=np.float32)

        # Frontend returns one feature frame per call
        mock_frontend = MagicMock()
        mock_frontend.process_samples.return_value = _make_mock_frontend_result(features=[0.5] * 40, samples_read=160)
        mock_frontend_cls.return_value = mock_frontend

        mww = MicroWakeWord(model_path="fake.tflite", sliding_window_size=3)

        # Feed enough chunks to fill feature buffer (3 slices needed, 3 features per 512-sample chunk)
        audio = np.zeros(512, dtype=np.int16)
        prob = mww.predict(audio)

        assert prob == pytest.approx(0.8)
        mock_interpreter.invoke.assert_called_once()

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_sliding_window_smoothing(self, mock_interpreter_cls, mock_frontend_cls):
        """Prediction window averages multiple inference results."""
        mock_interpreter = _make_mock_interpreter(input_feature_slices=2)
        mock_interpreter_cls.return_value = mock_interpreter

        mock_frontend = MagicMock()
        mock_frontend.process_samples.return_value = _make_mock_frontend_result(features=[0.5] * 40, samples_read=160)
        mock_frontend_cls.return_value = mock_frontend

        mww = MicroWakeWord(model_path="fake.tflite", sliding_window_size=3)
        audio = np.zeros(512, dtype=np.int16)

        # First prediction: 0.6
        mock_interpreter.get_tensor.return_value = np.array([[0.6]], dtype=np.float32)
        p1 = mww.predict(audio)
        assert p1 == pytest.approx(0.6)

        # Second prediction: 0.9 → average of [0.6, 0.9] = 0.75
        mock_interpreter.get_tensor.return_value = np.array([[0.9]], dtype=np.float32)
        p2 = mww.predict(audio)
        assert p2 == pytest.approx(0.75)

        # Third prediction: 0.3 → average of [0.6, 0.9, 0.3] = 0.6
        mock_interpreter.get_tensor.return_value = np.array([[0.3]], dtype=np.float32)
        p3 = mww.predict(audio)
        assert p3 == pytest.approx(0.6)


class TestMicroWakeWordCooldown:
    """Tests for cooldown / debounce behavior."""

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_cooldown_returns_zero_while_running_inference(self, mock_interpreter_cls, mock_frontend_cls):
        """During cooldown, predict returns 0.0 but inference runs to keep model state current."""
        mock_interpreter = _make_mock_interpreter(input_feature_slices=2)
        mock_interpreter_cls.return_value = mock_interpreter

        mock_frontend = MagicMock()
        mock_frontend.process_samples.return_value = _make_mock_frontend_result(features=[0.5] * 40, samples_read=160)
        mock_frontend_cls.return_value = mock_frontend

        mww = MicroWakeWord(model_path="fake.tflite", cooldown_chunks=3, rearm_threshold=0.9)
        audio = np.zeros(512, dtype=np.int16)

        # Fill feature buffer first
        mock_interpreter.get_tensor.return_value = np.array([[0.9]], dtype=np.float32)
        mww.predict(audio)

        # Activate cooldown
        mww.activate_cooldown()
        mock_interpreter.invoke.reset_mock()

        # During cooldown with low probability: returns 0.0 but DOES run inference
        mock_interpreter.get_tensor.return_value = np.array([[0.1]], dtype=np.float32)
        assert mww.predict(audio) == 0.0
        assert mww.predict(audio) == 0.0
        assert mww.predict(audio) == 0.0
        # Inference was called during cooldown (keeps model hidden state current)
        assert mock_interpreter.invoke.call_count == 3

        # After cooldown expires: detection re-enables
        mock_interpreter.get_tensor.return_value = np.array([[0.8]], dtype=np.float32)
        prob = mww.predict(audio)
        assert prob > 0.0

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_cooldown_only_decrements_on_low_probability(self, mock_interpreter_cls, mock_frontend_cls):
        """Cooldown stays active when model outputs probability >= rearm_threshold."""
        mock_interpreter = _make_mock_interpreter(input_feature_slices=2)
        mock_interpreter_cls.return_value = mock_interpreter

        mock_frontend = MagicMock()
        mock_frontend.process_samples.return_value = _make_mock_frontend_result(features=[0.5] * 40, samples_read=160)
        mock_frontend_cls.return_value = mock_frontend

        mww = MicroWakeWord(model_path="fake.tflite", cooldown_chunks=3, rearm_threshold=0.9)
        audio = np.zeros(512, dtype=np.int16)

        # Fill feature buffer first
        mock_interpreter.get_tensor.return_value = np.array([[0.9]], dtype=np.float32)
        mww.predict(audio)

        mww.activate_cooldown()
        assert mww._cooldown_remaining == 3

        # High probability during cooldown: cooldown does NOT decrement
        mock_interpreter.get_tensor.return_value = np.array([[0.95]], dtype=np.float32)
        for _ in range(5):
            assert mww.predict(audio) == 0.0
        assert mww._cooldown_remaining == 3  # Unchanged — model still "hot"

        # Low probability: cooldown starts decrementing
        mock_interpreter.get_tensor.return_value = np.array([[0.1]], dtype=np.float32)
        mww.predict(audio)
        assert mww._cooldown_remaining == 2

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_activate_cooldown_clears_prediction_window(self, mock_interpreter_cls, mock_frontend_cls):
        """activate_cooldown() clears the prediction smoothing window."""
        mock_interpreter = _make_mock_interpreter(input_feature_slices=2)
        mock_interpreter_cls.return_value = mock_interpreter
        mock_interpreter.get_tensor.return_value = np.array([[0.5]], dtype=np.float32)

        mock_frontend = MagicMock()
        mock_frontend.process_samples.return_value = _make_mock_frontend_result(features=[0.5] * 40, samples_read=160)
        mock_frontend_cls.return_value = mock_frontend

        mww = MicroWakeWord(model_path="fake.tflite", sliding_window_size=5, cooldown_chunks=1)
        audio = np.zeros(512, dtype=np.int16)

        # Build up prediction history
        mww.predict(audio)
        mww.predict(audio)
        assert len(mww._prediction_window) == 2

        mww.activate_cooldown()
        assert len(mww._prediction_window) == 0


class TestMicroWakeWordReset:
    """Tests for reset behavior."""

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_reset_clears_all_state(self, mock_interpreter_cls, mock_frontend_cls):
        mock_interpreter = _make_mock_interpreter(input_feature_slices=2)
        mock_interpreter_cls.return_value = mock_interpreter
        mock_interpreter.get_tensor.return_value = np.array([[0.5]], dtype=np.float32)

        mock_frontend = MagicMock()
        mock_frontend.process_samples.return_value = _make_mock_frontend_result(features=[0.5] * 40, samples_read=160)
        mock_frontend_cls.return_value = mock_frontend

        mww = MicroWakeWord(model_path="fake.tflite", cooldown_chunks=10)
        audio = np.zeros(512, dtype=np.int16)

        # Accumulate state
        mww.predict(audio)
        mww.predict(audio)
        assert len(mww._feature_buffer) > 0

        mww.activate_cooldown()
        assert mww._cooldown_remaining > 0

        mww.reset()

        assert len(mww._feature_buffer) == 0
        assert len(mww._prediction_window) == 0
        assert mww._cooldown_remaining == 0
        mock_frontend.reset.assert_called_once()


class TestMicroWakeWordFeatureBuffer:
    """Tests for feature ring buffer behavior."""

    @patch("private_assistant_comms_satellite.micro_wake_word.MicroFrontend")
    @patch("private_assistant_comms_satellite.micro_wake_word.Interpreter")
    def test_feature_buffer_bounded_by_maxlen(self, mock_interpreter_cls, mock_frontend_cls):
        """Feature buffer never exceeds input_feature_slices entries."""
        mock_interpreter = _make_mock_interpreter(input_feature_slices=3)
        mock_interpreter_cls.return_value = mock_interpreter
        mock_interpreter.get_tensor.return_value = np.array([[0.5]], dtype=np.float32)

        mock_frontend = MagicMock()
        mock_frontend.process_samples.return_value = _make_mock_frontend_result(features=[0.5] * 40, samples_read=160)
        mock_frontend_cls.return_value = mock_frontend

        mww = MicroWakeWord(model_path="fake.tflite")
        audio = np.zeros(512, dtype=np.int16)

        # Feed many chunks — buffer should stay bounded
        for _ in range(20):
            mww.predict(audio)

        assert len(mww._feature_buffer) <= 3
