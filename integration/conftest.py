"""Shared fixtures for integration tests."""

import math
import pathlib
import urllib.request

import numpy as np
import piper
import piper.download_voices as piper_download
import pytest
from scipy import signal

_OKAY_NABU_URL = "https://github.com/esphome/micro-wake-word-models/raw/main/models/v2/okay_nabu.tflite"
_PIPER_VOICE = "en_US-lessac-medium"
_TARGET_SAMPLE_RATE = 16_000


@pytest.fixture(scope="session")
def model_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Session-scoped directory for caching downloaded models."""
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def okay_nabu_model_path(model_cache_dir: pathlib.Path) -> pathlib.Path:
    """Download and cache the okay_nabu v2 TFLite model."""
    model_path = model_cache_dir / "okay_nabu.tflite"
    if not model_path.exists():
        urllib.request.urlretrieve(_OKAY_NABU_URL, model_path)
    return model_path


@pytest.fixture(scope="session")
def piper_voice(model_cache_dir: pathlib.Path) -> piper.PiperVoice:
    """Download and load a Piper TTS voice model."""
    piper_download.download_voice(_PIPER_VOICE, model_cache_dir)
    onnx_path = model_cache_dir / f"{_PIPER_VOICE}.onnx"
    return piper.PiperVoice.load(str(onnx_path))


def _resample(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Resample int16 audio using polyphase filtering."""
    if original_rate == target_rate:
        return audio
    gcd = math.gcd(target_rate, original_rate)
    resampled = signal.resample_poly(audio, target_rate // gcd, original_rate // gcd)
    return resampled.astype(np.int16)


@pytest.fixture(scope="session")
def okay_nabu_audio(piper_voice: piper.PiperVoice) -> np.ndarray:
    """Generate 'okay nabu' audio as int16 PCM at 16 kHz."""
    chunks = piper_voice.synthesize("okay nabu")
    audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in chunks)
    audio = np.frombuffer(audio_bytes, dtype=np.int16)

    voice_rate = piper_voice.config.sample_rate
    return _resample(audio, voice_rate, _TARGET_SAMPLE_RATE)
