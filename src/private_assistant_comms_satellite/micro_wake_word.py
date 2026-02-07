"""Streaming wake word detection using micro-wake-word TFLite models."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from pymicro_features import MicroFrontend

logger = logging.getLogger(__name__)

# MicroFrontend processes 160 samples (10ms at 16kHz) per call, as 16-bit PCM bytes
_SAMPLES_PER_FRAME = 160
_BYTES_PER_FRAME = _SAMPLES_PER_FRAME * 2


class MicroWakeWord:
    """Streaming wake word detector using TFLite and MicroFrontend.

    Generates spectrogram features incrementally from int16 PCM audio,
    maintains a ring buffer of features, runs TFLite inference when
    enough features accumulate, and applies sliding window smoothing.

    Args:
        model_path: Path to the .tflite wake word model file.
        sliding_window_size: Number of recent predictions to average for smoothing.
        cooldown_chunks: Minimum number of low-probability inferences required
            after activation before detection re-enables (matches ESPHome's
            ``ignore_windows`` mechanism).
        rearm_threshold: Raw probability must be below this value for a cooldown
            step to count. Keeps cooldown active while the model still outputs
            high probability. Should equal the detection threshold.

    """

    def __init__(
        self,
        model_path: str,
        sliding_window_size: int = 5,
        cooldown_chunks: int = 40,
        rearm_threshold: float = 0.9,
    ) -> None:
        """Initialize the wake word detector with a TFLite model."""
        self._interpreter = Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Input shape is [1, time_steps, num_features]
        self._input_feature_slices: int = int(self._input_details[0]["shape"][1])
        self._num_features: int = int(self._input_details[0]["shape"][2])

        # Quantization parameters for int8/uint8 models
        self._input_dtype = self._input_details[0]["dtype"]
        self._input_scale: float = float(self._input_details[0]["quantization"][0])
        self._input_zero_point: int = int(self._input_details[0]["quantization"][1])
        self._output_scale: float = float(self._output_details[0]["quantization"][0])
        self._output_zero_point: int = int(self._output_details[0]["quantization"][1])

        # Pre-compute clip bounds for quantized input
        if self._input_dtype != np.float32:
            info = np.iinfo(self._input_dtype)  # type: ignore[arg-type]
            self._input_clip_min: float = float(info.min)
            self._input_clip_max: float = float(info.max)

        self._frontend = MicroFrontend()

        # Ring buffer of spectrogram feature frames
        self._feature_buffer: deque[list[float]] = deque(maxlen=self._input_feature_slices)

        # Sliding window of recent predictions for smoothing
        self._prediction_window: deque[float] = deque(maxlen=sliding_window_size)

        # Cooldown / debounce — modelled after ESPHome's ignore_windows mechanism.
        # After detection the model keeps running inference so its hidden state
        # stays current, but detection is suppressed until *cooldown_chunks*
        # consecutive low-probability inferences have been observed.
        self._cooldown_total = cooldown_chunks
        self._cooldown_remaining = 0
        self._rearm_threshold = rearm_threshold

        logger.info(
            "MicroWakeWord loaded: %d feature slices x %d features, sliding_window=%d, cooldown=%d",
            self._input_feature_slices,
            self._num_features,
            sliding_window_size,
            cooldown_chunks,
        )

    def predict(self, audio_int16: np.ndarray) -> float:
        """Process an int16 PCM audio chunk and return smoothed wake word probability.

        Args:
            audio_int16: Audio samples as int16 numpy array (any length,
                typically 512 samples / 32ms at 16kHz).

        Returns:
            Smoothed probability between 0.0 and 1.0. Returns 0.0 during cooldown
            or when insufficient features have accumulated.

        """
        audio_bytes = audio_int16.tobytes()
        num_bytes = len(audio_bytes)
        byte_idx = 0

        while byte_idx + _BYTES_PER_FRAME <= num_bytes:
            result = self._frontend.process_samples(audio_bytes[byte_idx : byte_idx + _BYTES_PER_FRAME])
            byte_idx += result.samples_read * 2
            if result.features:
                self._feature_buffer.append(result.features)

        if len(self._feature_buffer) < self._input_feature_slices:
            return 0.0

        # Always run inference so the model's internal hidden state stays
        # current with the audio stream (v2 models are streaming models that
        # accumulate temporal context through TFLite resource variables).
        spectrogram = np.array(list(self._feature_buffer), dtype=np.float32)

        # Quantize if model expects integer input (v2 models use int8)
        if self._input_dtype != np.float32:
            spectrogram = np.clip(
                np.round(spectrogram / self._input_scale) + self._input_zero_point,
                self._input_clip_min,
                self._input_clip_max,
            ).astype(self._input_dtype)

        input_tensor = spectrogram.reshape(self._input_details[0]["shape"])
        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()
        raw_output = self._interpreter.get_tensor(self._output_details[0]["index"])

        # Dequantize if model outputs integer (v2 models use uint8)
        if self._output_details[0]["dtype"] != np.float32:
            raw_probability = (float(raw_output[0][0]) - self._output_zero_point) * self._output_scale
        else:
            raw_probability = float(raw_output[0][0])

        # During cooldown, inference still ran (above) to keep hidden state
        # fresh, but we suppress detection.  Only count down when the model
        # outputs low probability — this ensures the model has genuinely
        # "cooled off" before re-arming (matches ESPHome's ignore_windows).
        if self._cooldown_remaining > 0:
            if raw_probability < self._rearm_threshold:
                self._cooldown_remaining -= 1
            return 0.0

        self._prediction_window.append(raw_probability)
        return sum(self._prediction_window) / len(self._prediction_window)

    def activate_cooldown(self) -> None:
        """Activate cooldown period to prevent repeated detections."""
        self._cooldown_remaining = self._cooldown_total
        self._prediction_window.clear()

    def reset(self) -> None:
        """Reset all internal state."""
        self._feature_buffer.clear()
        self._prediction_window.clear()
        self._cooldown_remaining = 0
        self._frontend.reset()
