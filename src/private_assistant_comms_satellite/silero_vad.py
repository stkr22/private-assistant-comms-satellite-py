"""Voice activity detection."""

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector


class SileroVad:
    """Voice activity detection with silero VAD."""

    def __init__(self, threshold: float, trigger_level: int) -> None:
        self.detector = SileroVoiceActivityDetector()
        self.threshold = threshold
        self.trigger_level = trigger_level
        self._activation = 0

        # AIDEV-NOTE: Fixed chunk size for process_samples method (512 float samples)
        self._chunk_samples = 512

    def __call__(self, audio_array: np.ndarray | None) -> bool:
        """Process audio chunk and detect voice activity with trigger mechanism.

        Args:
            audio_array: Audio data as float32 array in range [-1, 1] or None to reset detector

        Returns:
            True if sustained speech detected above trigger level
        """
        # AIDEV-NOTE: VAD reset mechanism for initialization and error recovery
        if audio_array is None:
            # Reset
            self._activation = 0
            self.detector.reset()
            return False

        if len(audio_array) < self._chunk_samples:
            raise ValueError(f"Audio array must be at least {self._chunk_samples} samples")

        # AIDEV-NOTE: Direct processing with new process_array method
        return self._process_audio_direct(audio_array)

    def _process_audio_direct(self, audio_array: np.ndarray) -> bool:
        """Direct audio processing using new process_array method."""
        audio_len = len(audio_array)

        # AIDEV-NOTE: Calculate number of complete chunks to process
        num_chunks = audio_len // self._chunk_samples

        if num_chunks == 0:
            return False

        # AIDEV-NOTE: Optimize for single chunk (most common case)
        if num_chunks == 1:
            # Use audio directly - process_samples accepts [-1, 1] range
            chunk = audio_array[: self._chunk_samples]
            prob = self.detector.process_samples(chunk.tolist())
            return self._update_activation(prob)

        # AIDEV-NOTE: Process multiple chunks efficiently
        max_prob = 0.0

        for i in range(num_chunks):
            start_idx = i * self._chunk_samples
            end_idx = start_idx + self._chunk_samples

            # Use audio directly - process_samples accepts [-1, 1] range
            chunk = audio_array[start_idx:end_idx]
            prob = self.detector.process_samples(chunk.tolist())
            max_prob = max(max_prob, prob)

            # Early termination if we already found high probability
            if max_prob >= self.threshold:
                break

        return self._update_activation(max_prob)

    def _update_activation(self, probability: float) -> bool:
        """Update activation state based on probability."""
        # AIDEV-NOTE: Trigger-level mechanism reduces false positives in edge device environments
        if probability >= self.threshold:
            # Speech detected
            self._activation += 1
            if self._activation >= self.trigger_level:
                self._activation = 0
                return True
        else:
            # Silence detected
            self._activation = max(0, self._activation - 1)

        return False
