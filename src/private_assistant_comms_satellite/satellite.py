from __future__ import annotations

import asyncio
import enum
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from private_assistant_commons import messages

# AIDEV-NOTE: Import pyaudio as optional dependency for CI/CD compatibility
if TYPE_CHECKING:
    import logging

    import openwakeword
    import pyaudio
else:
    try:
        import pyaudio
    except ImportError:
        pyaudio = None

from private_assistant_comms_satellite import silero_vad
from private_assistant_comms_satellite.utils import (
    config,
    mqtt_utils,
    speech_recognition_tools,
)


class SatelliteState(enum.Enum):
    """Satellite operation states."""

    LISTENING = "listening"  # Analyzing audio for wakeword
    RECORDING = "recording"  # Recording user command after wakeword
    WAITING = "waiting"  # Waiting for API response
    SPEAKING = "speaking"  # Playing TTS audio


class Satellite:
    """Main satellite class for voice interaction with simple state machine."""

    def __init__(  # noqa: PLR0913
        self,
        config: config.Config,
        output_queue: asyncio.Queue[messages.Response],
        start_listening_sound: bytes,
        stop_listening_sound: bytes,
        wakeword_model: openwakeword.Model,
        mqtt_client: mqtt_utils.AsyncMQTTClient,
        logger: logging.Logger,
    ):
        # Assign configuration
        self.config = config
        self.output_queue = output_queue
        # Assign preloaded WAV data
        self.start_listening_sound: bytes = start_listening_sound
        self.stop_listening_sound: bytes = stop_listening_sound
        self.vad_model = silero_vad.SileroVad(threshold=config.vad_threshold, trigger_level=config.vad_trigger)
        self.logger = logger
        # Assign wakeword model
        self.wakeword_model: openwakeword.Model = wakeword_model
        self.mqtt_client = mqtt_client
        self._mqtt_thread: threading.Thread | None = None

        # AIDEV-NOTE: Simple state machine
        self._state = SatelliteState.LISTENING
        self._state_lock = threading.RLock()

        # AIDEV-NOTE: Audio recording buffer
        max_recording_samples = self.config.max_command_input_seconds * self.config.samplerate
        self._audio_buffer = np.zeros(max_recording_samples, dtype=np.float32)
        self._buffer_position = 0

        # AIDEV-NOTE: PyAudio initialization - critical dependency for edge device audio I/O
        if pyaudio is None:
            raise ImportError("PyAudio is required for audio processing. Install with: uv sync --group audio")

        self.p: Any = pyaudio.PyAudio()
        self.stream_input: Any = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.samplerate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            input_device_index=self.config.input_device_index,
        )
        self.stream_output: Any = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.samplerate,
            output=True,
            output_device_index=self.config.output_device_index,
        )

    def _get_state(self) -> SatelliteState:
        """Thread-safe state getter."""
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: SatelliteState) -> None:
        """Thread-safe state setter."""
        with self._state_lock:
            if self._state != new_state:
                self.logger.info("State change: %s -> %s", self._state.value, new_state.value)
                self._state = new_state

    async def _process_output_queue(self) -> None:
        """Process pending TTS responses from the assistant."""
        try:
            response = self.output_queue.get_nowait()
            self.logger.info("Received new message: '%s'", response.text)

            # AIDEV-NOTE: Set state to waiting, then speaking
            self._set_state(SatelliteState.WAITING)

            # Get TTS audio synchronously
            audio_bytes = await speech_recognition_tools.send_text_to_tts_api(response.text, self.config)

            if audio_bytes:
                self._set_state(SatelliteState.SPEAKING)
                # Play audio directly
                self.stream_output.write(audio_bytes)
                self.logger.info("Finished playing TTS audio")

            # Return to listening state
            self._set_state(SatelliteState.LISTENING)

        except asyncio.QueueEmpty:
            pass  # No messages to process

    def _reset_audio_buffer(self) -> None:
        """Reset the audio buffer for new recording."""
        self._buffer_position = 0

    def _append_to_audio_buffer(self, audio_chunk: np.ndarray) -> bool:
        """Append audio chunk to buffer. Returns False if buffer is full."""
        chunk_size = len(audio_chunk)
        if self._buffer_position + chunk_size > len(self._audio_buffer):
            return False  # Buffer full

        self._audio_buffer[self._buffer_position : self._buffer_position + chunk_size] = audio_chunk
        self._buffer_position += chunk_size
        return True

    def _get_recorded_audio(self) -> np.ndarray:
        """Get the recorded audio."""
        if self._buffer_position == 0:
            return np.array([], dtype=np.float32)
        return self._audio_buffer[: self._buffer_position].copy()

    async def _record_command(self) -> None:
        """Record voice command after wake word detection."""
        self._set_state(SatelliteState.RECORDING)
        self._reset_audio_buffer()

        try:
            silence_packages = 0
            max_frames = self.config.max_command_input_seconds * self.config.samplerate
            max_silent_packages = self.config.samplerate / self.config.chunk_size * self.config.max_length_speech_pause
            has_audio_data = False
            active_listening = True

            while active_listening:
                # Read directly from audio stream
                audio_bytes = self.stream_input.read(self.config.chunk_size, exception_on_overflow=False)

                if self.vad_model(audio_bytes) > self.config.vad_threshold:
                    # Convert audio to float32
                    raw_audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_chunk = raw_audio_data.astype(np.float32) / 32768.0

                    silence_packages = 0
                    if not self._append_to_audio_buffer(audio_chunk):
                        active_listening = False
                        self.logger.warning("Audio buffer full, stopping recording")
                    else:
                        has_audio_data = True
                    self.logger.debug("Received voice...")
                else:
                    if has_audio_data:
                        raw_audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_chunk = raw_audio_data.astype(np.float32) / 32768.0

                        if not self._append_to_audio_buffer(audio_chunk):
                            active_listening = False
                            self.logger.warning("Audio buffer full, stopping recording")
                        else:
                            silence_packages += 1
                    self.logger.debug("No voice...")

                if has_audio_data and (self._buffer_position > max_frames or silence_packages >= max_silent_packages):
                    active_listening = False
                    self.logger.info("Stopping listening, playing stop sound...")
                    self.stream_output.write(self.stop_listening_sound)

                    # Process STT
                    audio_frames = self._get_recorded_audio()
                    if len(audio_frames) > 0:
                        await self._process_stt(audio_frames)

                await asyncio.sleep(0.001)

        finally:
            self._set_state(SatelliteState.LISTENING)

    async def _process_stt(self, audio_frames: np.ndarray) -> None:
        """Process speech-to-text and publish to MQTT."""
        self._set_state(SatelliteState.WAITING)
        self.logger.info("Processing STT request")

        try:
            response = await speech_recognition_tools.send_audio_to_stt_api(audio_frames, config_obj=self.config)

            if response is not None:
                self.logger.info("STT result: '%s'", response.text)

                message = messages.ClientRequest(
                    id=uuid.uuid4(),
                    text=response.text,
                    room=self.config.room,
                    output_topic=self.config.output_topic,
                ).model_dump_json()

                await self.mqtt_client.publish(self.config.input_topic, message, qos=1)
                self.logger.info("Published result text to MQTT.")
            else:
                self.logger.warning("STT processing failed for recorded audio")
        except Exception as e:
            self.logger.error("STT processing failed: %s", e)
        finally:
            self._set_state(SatelliteState.LISTENING)

    def _start_mqtt_loop(self) -> None:
        """Start the MQTT client in a separate thread."""

        def mqtt_runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.mqtt_client.start())
            except Exception as e:
                self.logger.error("MQTT client error: %s", e)
            finally:
                loop.close()

        self._mqtt_thread = threading.Thread(target=mqtt_runner, daemon=True)
        self._mqtt_thread.start()
        time.sleep(1)  # Give MQTT client time to start

    async def _main_loop(self) -> None:
        """Main processing loop - handles wake word detection and state transitions."""
        try:
            while True:
                current_state = self._get_state()

                # Only process audio for wake word in LISTENING state
                if current_state == SatelliteState.LISTENING:
                    # Read audio chunk optimized for wake word detection
                    audio_data = self.stream_input.read(self.config.chunk_size_ow, exception_on_overflow=False)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)

                    # Wake word detection
                    prediction = self.wakeword_model.predict(
                        audio_np,
                        debounce_time=2.0,
                        threshold={self.config.name_wakeword_model: self.config.wakework_detection_threshold},
                    )
                    wakeword_probability = prediction[self.config.name_wakeword_model]

                    self.logger.debug(
                        "Wakeword probability: %s, Threshold: %s",
                        wakeword_probability,
                        self.config.wakework_detection_threshold,
                    )

                    if wakeword_probability >= self.config.wakework_detection_threshold:
                        self.logger.info("Wakeword detected, playing start listening sound.")
                        self.stream_output.write(self.start_listening_sound)
                        await self._record_command()

                # Always check for TTS messages (low latency MQTT processing)
                await self._process_output_queue()

                # Small yield to prevent busy-waiting
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            raise

    def start(self) -> None:
        """Start the satellite with simple state machine."""
        # Start MQTT client in separate thread
        self._start_mqtt_loop()

        # Run main processing loop
        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            raise

    def cleanup(self) -> None:
        """Clean up resources including MQTT client and audio streams."""
        if self._mqtt_thread and self._mqtt_thread.is_alive():
            self._mqtt_thread.join(timeout=2.0)

        self.stream_input.stop_stream()
        self.stream_output.stop_stream()
        self.stream_input.close()
        self.stream_output.close()
        self.p.terminate()
