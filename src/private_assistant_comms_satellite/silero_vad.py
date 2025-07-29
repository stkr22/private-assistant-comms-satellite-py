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

        # AIDEV-NOTE: Pre-calculate chunk size and optimize for vectorized processing
        self._chunk_size = self.detector.chunk_bytes()
        self._chunk_array = np.zeros(self._chunk_size, dtype=np.uint8)  # Reusable chunk buffer

    def __call__(self, audio_bytes: bytes | None) -> bool:
        """Process audio chunk and detect voice activity with trigger mechanism.

        Args:
            audio_bytes: Audio data chunk or None to reset detector

        Returns:
            True if sustained speech detected above trigger level
        """
        # AIDEV-NOTE: VAD reset mechanism for initialization and error recovery
        if audio_bytes is None:
            # Reset
            self._activation = 0
            self.detector.reset()
            return False

        if len(audio_bytes) < self._chunk_size:
            raise ValueError(f"Audio bytes must be at least {self._chunk_size} bytes")

        # AIDEV-NOTE: Vectorized processing - process larger chunks efficiently
        return self._process_audio_vectorized(audio_bytes)

    def _process_audio_vectorized(self, audio_bytes: bytes) -> bool:
        """Vectorized audio processing for improved performance."""
        audio_len = len(audio_bytes)

        # AIDEV-NOTE: Calculate number of complete chunks to process
        num_chunks = audio_len // self._chunk_size

        if num_chunks == 0:
            return False

        # AIDEV-NOTE: Optimize for single chunk (most common case)
        if num_chunks == 1:
            prob = self.detector(audio_bytes[: self._chunk_size])
            return self._update_activation(prob)

        # AIDEV-NOTE: Vectorized processing for multiple chunks
        # Convert bytes to numpy array for efficient slicing
        audio_array = np.frombuffer(audio_bytes, dtype=np.uint8)

        # Process chunks in batches to reduce Python loop overhead
        batch_size = min(8, num_chunks)  # Process up to 8 chunks at once
        max_prob = 0.0

        for batch_start in range(0, num_chunks, batch_size):
            batch_end = min(batch_start + batch_size, num_chunks)
            batch_probs = []

            # Process batch of chunks
            for i in range(batch_start, batch_end):
                start_idx = i * self._chunk_size
                end_idx = start_idx + self._chunk_size

                # AIDEV-NOTE: Use array view instead of bytes slicing for efficiency
                chunk_view = audio_array[start_idx:end_idx]
                self._chunk_array[:] = chunk_view  # Copy to reusable buffer
                prob = self.detector(self._chunk_array.tobytes())
                batch_probs.append(prob)

            # AIDEV-NOTE: Use numpy max for efficient batch processing
            batch_max = np.max(batch_probs) if batch_probs else 0.0
            max_prob = max(max_prob, batch_max)

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
