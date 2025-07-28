"""Voice activity detection."""

from pysilero_vad import SileroVoiceActivityDetector


class SileroVad:
    """Voice activity detection with silero VAD."""

    def __init__(self, threshold: float, trigger_level: int) -> None:
        self.detector = SileroVoiceActivityDetector()
        self.threshold = threshold
        self.trigger_level = trigger_level
        self._activation = 0

    def __call__(self, audio_bytes: bytes | None) -> bool:
        if audio_bytes is None:
            # Reset
            self._activation = 0
            self.detector.reset()
            return False

        chunk_size = self.detector.chunk_bytes()
        if len(audio_bytes) < chunk_size:
            raise ValueError(f"Audio bytes must be at least {chunk_size} bytes")

        # Process chunks
        speech_probs = []
        for i in range(0, len(audio_bytes) - chunk_size + 1, chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            speech_probs.append(self.detector(chunk))

        # Use maximum probability
        max_prob = max(speech_probs)

        if max_prob >= self.threshold:
            # Speech detected
            self._activation += 1
            if self._activation >= self.trigger_level:
                self._activation = 0
                return True
        else:
            # Silence detected
            self._activation = max(0, self._activation - 1)

        return False
