import asyncio
import logging
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import openwakeword
from private_assistant_commons import messages

# AIDEV-NOTE: Import pyaudio as optional dependency for CI/CD compatibility
if TYPE_CHECKING:
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


class Satellite:
    """Main satellite class for voice interaction."""

    # AIDEV-NOTE: Constructor has many parameters but they're all necessary for proper initialization
    def __init__(  # noqa: PLR0913
        self,
        config: config.Config,
        output_queue: queue.Queue[str],
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
        self._mqtt_loop: asyncio.AbstractEventLoop | None = None
        self._mqtt_thread: threading.Thread | None = None

        # Initialize PyAudio (if available)
        if pyaudio is None:
            raise ImportError(
                "PyAudio is required for audio processing. Install with: uv sync --group audio"
            )
        
        # AIDEV-NOTE: Use Any for runtime type flexibility with optional pyaudio
        
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

    def process_output_queue(self):
        try:
            output_text = self.output_queue.get_nowait()
            self.logger.info("Received new message: '%s'", output_text)
            self.logger.info("...requesting synthesize...")

            # AIDEV-NOTE: Run async TTS call in the MQTT event loop
            if self._mqtt_loop and not self._mqtt_loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    speech_recognition_tools.send_text_to_tts_api(output_text, self.config), self._mqtt_loop
                )
                try:
                    # Wait for TTS result with timeout
                    audio_bytes = future.result(timeout=10.0)
                    if audio_bytes is not None:
                        self.stream_output.write(audio_bytes)
                except Exception as e:
                    self.logger.error("TTS synthesis failed: %s", e)
            else:
                self.logger.error("MQTT loop not available, cannot process TTS")
        except queue.Empty:
            self.logger.debug("Queue is empty, no message to process.")

    def processing_spoken_commands(self) -> None:
        silence_packages = 0
        max_frames = self.config.max_command_input_seconds * self.config.samplerate
        max_silent_packages = self.config.samplerate / self.config.chunk_size * self.config.max_length_speech_pause
        audio_frames = None
        active_listening = True
        while active_listening:
            audio_bytes = self.stream_input.read(self.config.chunk_size, exception_on_overflow=False)
            if self.vad_model(audio_bytes) > self.config.vad_threshold:
                raw_audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                data = speech_recognition_tools.int2float(raw_audio_data)
                silence_packages = 0
                audio_frames = data if audio_frames is None else np.concatenate((audio_frames, data), axis=0)
                self.logger.debug("Received voice...")
            else:
                if audio_frames is not None:
                    audio_frames = np.concatenate((audio_frames, data), axis=0)
                    silence_packages += 1
                self.logger.debug("No voice...")
            if audio_frames is not None and (
                audio_frames.shape[0] > max_frames or silence_packages >= max_silent_packages
            ):
                active_listening = False
                self.logger.info("Stopping listening, playing stop sound...")
                self.stream_output.write(self.stop_listening_sound)
                self.logger.info("Requested transcription...")

                # AIDEV-NOTE: Run async STT call in the MQTT event loop
                if self._mqtt_loop and not self._mqtt_loop.is_closed():
                    future = asyncio.run_coroutine_threadsafe(
                        speech_recognition_tools.send_audio_to_stt_api(audio_frames, config_obj=self.config),
                        self._mqtt_loop,
                    )
                    try:
                        # Wait for STT result with timeout
                        response = future.result(timeout=15.0)
                        self.logger.info("Received result...%s", response)

                        if response is not None:
                            message = messages.ClientRequest(
                                id=uuid.uuid4(),
                                text=response.text,
                                room=self.config.room,
                                output_topic=self.config.output_topic,
                            ).model_dump_json()

                            asyncio.run_coroutine_threadsafe(
                                self.mqtt_client.publish(self.config.input_topic, message, qos=1), self._mqtt_loop
                            )
                            self.logger.info("Published result text to MQTT.")
                    except Exception as e:
                        self.logger.error("STT transcription failed: %s", e)
                else:
                    self.logger.error("MQTT loop not available, cannot process STT")

    def _start_mqtt_loop(self) -> None:
        """Start the MQTT client in a separate thread with its own event loop."""

        # AIDEV-NOTE: Run MQTT client in separate thread to avoid blocking audio processing
        def mqtt_runner():
            self._mqtt_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._mqtt_loop)
            try:
                self._mqtt_loop.run_until_complete(self.mqtt_client.start())
            except Exception as e:
                self.logger.error("MQTT client error: %s", e)
            finally:
                self._mqtt_loop.close()

        self._mqtt_thread = threading.Thread(target=mqtt_runner, daemon=True)
        self._mqtt_thread.start()
        # Give the MQTT client time to start
        time.sleep(1)

    def start(self) -> None:
        """Start the satellite with MQTT client and audio processing."""
        self._start_mqtt_loop()

        try:
            while True:
                self.process_output_queue()
                audio_data = self.stream_input.read(self.config.chunk_size_ow, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)

                # Wakeword detection
                prediction = self.wakeword_model.predict(
                    audio_np,
                    debounce_time=2.0,
                    threshold={self.config.name_wakeword_model: self.config.wakework_detection_threshold},
                )
                wakeword_probability = prediction[self.config.name_wakeword_model]

                self.logger.debug(
                    "Wakeword probability: %s, Threshold check: %s",
                    wakeword_probability,
                    wakeword_probability >= self.config.wakework_detection_threshold,
                )

                if wakeword_probability >= self.config.wakework_detection_threshold:
                    self.logger.info("Wakeword detected, playing start listening sound.")
                    self.stream_output.write(self.start_listening_sound)
                    self.processing_spoken_commands()
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            raise

    def cleanup(self) -> None:
        """Clean up resources including MQTT client and audio streams."""
        # AIDEV-NOTE: Stop MQTT client asynchronously and clean up threading
        if self._mqtt_loop and not self._mqtt_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.mqtt_client.stop(), self._mqtt_loop)

        if self._mqtt_thread and self._mqtt_thread.is_alive():
            self._mqtt_thread.join(timeout=2.0)

        self.stream_input.stop_stream()
        self.stream_output.stop_stream()
        self.stream_input.close()
        self.stream_output.close()
        self.p.terminate()
