"""Integration tests for MicroWakeWord with real TFLite model.

Uses Piper TTS to synthesize wake word audio and feeds it through the
real okay_nabu.tflite model to verify detection and cooldown behaviour.
"""

import pathlib

import numpy as np
import pytest

from private_assistant_comms_satellite.micro_wake_word import MicroWakeWord

_CHUNK_SIZE = 512
_DETECTION_THRESHOLD = 0.97
_RELAXED_THRESHOLD = 0.5  # TTS output may not perfectly match natural speech


@pytest.fixture()
def detector(okay_nabu_model_path: pathlib.Path) -> MicroWakeWord:
    """Create a MicroWakeWord instance with the real okay_nabu model."""
    return MicroWakeWord(
        model_path=str(okay_nabu_model_path),
        sliding_window_size=5,
        cooldown_chunks=40,
        rearm_threshold=_DETECTION_THRESHOLD,
    )


def _feed_audio(detector: MicroWakeWord, audio: np.ndarray) -> list[float]:
    """Feed audio in chunks and return all probabilities."""
    probabilities: list[float] = []
    for i in range(0, len(audio), _CHUNK_SIZE):
        chunk = audio[i : i + _CHUNK_SIZE]
        if len(chunk) < _CHUNK_SIZE:
            chunk = np.pad(chunk, (0, _CHUNK_SIZE - len(chunk)))
        probabilities.append(detector.predict(chunk))
    return probabilities


@pytest.mark.integration
def test_no_false_positive_on_silence(detector: MicroWakeWord) -> None:
    """Feeding silence should never exceed the detection threshold."""
    silence = np.zeros(16_000 * 5, dtype=np.int16)  # 5 seconds
    probabilities = _feed_audio(detector, silence)
    assert all(p < _DETECTION_THRESHOLD for p in probabilities), (
        f"False positive on silence: max={max(probabilities):.4f}"
    )


@pytest.mark.integration
def test_detects_wake_word(
    detector: MicroWakeWord,
    okay_nabu_audio: np.ndarray,
) -> None:
    """Piper-synthesized 'okay nabu' should be detected by the model."""
    # Warm up with a short silence preamble
    warmup = np.zeros(16_000, dtype=np.int16)  # 1 second
    _feed_audio(detector, warmup)

    probabilities = _feed_audio(detector, okay_nabu_audio)
    max_prob = max(probabilities)
    assert max_prob > _RELAXED_THRESHOLD, f"Model did not respond to TTS wake word: max={max_prob:.4f}"


@pytest.mark.integration
def test_no_retrigger_after_detection(
    detector: MicroWakeWord,
    okay_nabu_audio: np.ndarray,
) -> None:
    """After detection + cooldown activation, silence must not re-trigger."""
    # Warm up
    warmup = np.zeros(16_000, dtype=np.int16)
    _feed_audio(detector, warmup)

    # Feed wake word audio until detection (or finish the clip)
    detected = False
    for i in range(0, len(okay_nabu_audio), _CHUNK_SIZE):
        chunk = okay_nabu_audio[i : i + _CHUNK_SIZE]
        if len(chunk) < _CHUNK_SIZE:
            chunk = np.pad(chunk, (0, _CHUNK_SIZE - len(chunk)))
        prob = detector.predict(chunk)
        if prob >= _DETECTION_THRESHOLD:
            detected = True
            detector.activate_cooldown()
            break

    if not detected:
        pytest.skip("TTS audio did not trigger detection — cannot test re-trigger")

    # Feed 10 seconds of silence after cooldown activation
    silence = np.zeros(16_000 * 10, dtype=np.int16)
    probabilities = _feed_audio(detector, silence)
    max_prob = max(probabilities) if probabilities else 0.0
    assert max_prob < _DETECTION_THRESHOLD, f"Re-triggered after cooldown: max={max_prob:.4f}"
