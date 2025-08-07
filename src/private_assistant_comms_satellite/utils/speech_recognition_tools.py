import logging

import httpx
import numpy as np
import numpy.typing as np_typing
from pydantic import BaseModel, ValidationError

from private_assistant_comms_satellite.utils import (
    config,
)

logger = logging.getLogger(__name__)


class STTResponse(BaseModel):
    text: str
    message: str


async def send_audio_to_stt_api(
    audio_data: np_typing.NDArray[np.float32],
    config_obj: config.Config,
    timeout: float = 10.0,
) -> STTResponse | None:
    """Send audio to STT API and receive transcription.

    Args:
        audio_data: Float32 audio array to transcribe
        config_obj: Configuration with API endpoint and token
        timeout: Request timeout in seconds

    Returns:
        STTResponse with transcribed text or None if failed
    """
    # AIDEV-NOTE: Async API call executed in MQTT thread to avoid blocking audio processing
    files = {"file": ("audio.raw", audio_data.tobytes())}
    headers = {"user-token": config_obj.speech_transcription_api_token or ""}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                config_obj.speech_transcription_api,
                files=files,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return STTResponse.model_validate(response.json())

    except httpx.TimeoutException:
        logger.error("Request timed out after %.1f seconds", timeout)
    except httpx.HTTPStatusError as e:
        logger.error("HTTP %d error: %s", e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        logger.error("Network error: %s", e)
    except ValidationError as e:
        logger.error("Response validation error: %s", e.errors())

    return None


async def send_text_to_tts_api(
    text: str,
    config_obj: config.Config,
    sample_rate: int = 16000,
    timeout: float = 10.0,
) -> bytes | None:
    """Send text to TTS API and receive audio data.

    Args:
        text: Text to synthesize into speech
        config_obj: Configuration with API endpoint and token
        sample_rate: Audio sample rate for synthesis
        timeout: Request timeout in seconds

    Returns:
        Audio bytes in WAV format or None if failed
    """
    # AIDEV-NOTE: Async API call with error handling for network resilience
    headers = {
        "user-token": config_obj.speech_synthesis_api_token or "",
        "Content-Type": "application/json",
    }

    payload = {"text": text, "sample_rate": sample_rate}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=config_obj.speech_synthesis_api,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            min_audio_bytes = 2
            if len(response.content) < min_audio_bytes:
                logger.error("Insufficient audio data: %d bytes", len(response.content))
                return None

            return response.content

    except httpx.TimeoutException:
        logger.error("Request timed out after %.1f seconds", timeout)
    except httpx.HTTPStatusError as e:
        logger.error("HTTP %d error: %s", e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        logger.error("Network error: %s", e)
    except ValueError as e:
        logger.error("Audio conversion error: %s", e)

    return None
