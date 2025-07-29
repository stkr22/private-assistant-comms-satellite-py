from __future__ import annotations

import asyncio
import queue
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


class Satellite:
    """Main satellite class for voice interaction."""

    # AIDEV-NOTE: Constructor has many parameters but they're all necessary for proper initialization
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
        self._mqtt_loop: asyncio.AbstractEventLoop | None = None
        self._mqtt_thread: threading.Thread | None = None

        # AIDEV-NOTE: Queues for non-blocking API communication
        self._tts_request_queue: queue.Queue[str] = queue.Queue()
        self._tts_response_queue: queue.Queue[bytes | None] = queue.Queue()
        self._stt_request_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stt_response_queue: queue.Queue[str | None] = queue.Queue()

        # AIDEV-NOTE: PyAudio initialization - critical dependency for edge device audio I/O
        if pyaudio is None:
            raise ImportError("PyAudio is required for audio processing. Install with: uv sync --group audio")

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

    async def process_output_queue(self):
        """Process pending TTS responses from the assistant.

        Checks output queue for new messages and queues them for non-blocking TTS processing.
        Plays any completed TTS audio that's ready. Called frequently in main loop.
        """
        # AIDEV-NOTE: Critical performance path - non-blocking queue operations only
        try:
            response = self.output_queue.get_nowait()
            self.logger.info("Received new message: '%s'", response.text)
            self.logger.info("...queuing for TTS synthesis...")

            # AIDEV-NOTE: Non-blocking - just queue the request
            self._tts_request_queue.put(response.text)
        except queue.Empty:
            self.logger.debug("Queue is empty, no message to process.")

        # AIDEV-NOTE: Check for completed TTS responses (non-blocking)
        try:
            audio_bytes = self._tts_response_queue.get_nowait()
            if audio_bytes is not None:
                self.stream_output.write(audio_bytes)
                self.logger.info("Played TTS audio")
        except queue.Empty:
            pass  # No completed TTS responses ready

    async def processing_spoken_commands(self) -> None:
        """Handle voice command recording after wake word detection.

        Records audio until silence detected or timeout, uses VAD for speech boundaries,
        calls STT API asynchronously, and publishes transcription to MQTT.
        """
        # AIDEV-NOTE: Core audio processing loop - handles real-time recording and VAD
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

                # AIDEV-NOTE: Non-blocking - just queue the audio for STT processing
                self._stt_request_queue.put(audio_frames)

                # AIDEV-NOTE: Continue audio processing while STT happens in background
                # We'll check for STT results in the main loop

            # AIDEV-NOTE: Allow other async tasks to run during voice recording
            await asyncio.sleep(0.001)

    def _api_processor(self) -> None:
        """Background thread for processing API requests without blocking audio."""

        async def process_api_requests():
            """Async processing of TTS and STT requests."""
            while True:
                # Process TTS requests
                try:
                    text = self._tts_request_queue.get_nowait()
                    self.logger.info("Processing TTS request: '%s'", text)
                    audio_bytes = await speech_recognition_tools.send_text_to_tts_api(text, self.config)
                    self._tts_response_queue.put(audio_bytes)
                    self.logger.info("TTS request completed")
                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error("TTS processing failed: %s", e)
                    self._tts_response_queue.put(None)

                # Process STT requests
                try:
                    audio_frames = self._stt_request_queue.get_nowait()
                    self.logger.info("Processing STT request")
                    response = await speech_recognition_tools.send_audio_to_stt_api(
                        audio_frames, config_obj=self.config
                    )
                    if response is not None:
                        self._stt_response_queue.put(response.text)
                        self.logger.info("STT request completed: '%s'", response.text)
                    else:
                        self._stt_response_queue.put(None)
                        self.logger.warning("STT request failed")
                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error("STT processing failed: %s", e)
                    self._stt_response_queue.put(None)

                # Small delay to avoid busy-waiting
                await asyncio.sleep(0.01)

        # AIDEV-NOTE: Run API processing in separate thread with its own event loop
        api_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(api_loop)
        try:
            api_loop.run_until_complete(process_api_requests())
        except Exception as e:
            self.logger.error("API processor error: %s", e)
        finally:
            api_loop.close()

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

    async def _check_stt_results(self) -> None:
        """Check for completed STT results and publish to MQTT (non-blocking)."""
        try:
            result_text = self._stt_response_queue.get_nowait()
            if result_text is not None:
                self.logger.info("Received STT result: '%s'", result_text)

                message = messages.ClientRequest(
                    id=uuid.uuid4(),
                    text=result_text,
                    room=self.config.room,
                    output_topic=self.config.output_topic,
                ).model_dump_json()

                if self._mqtt_loop and not self._mqtt_loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self.mqtt_client.publish(self.config.input_topic, message, qos=1), self._mqtt_loop
                    )
                    self.logger.info("Published result text to MQTT.")
                else:
                    self.logger.error("MQTT loop not available, cannot publish STT result")
            else:
                self.logger.warning("STT processing failed for recorded audio")
        except queue.Empty:
            pass  # No STT results ready yet

    async def _audio_processing_loop(self) -> None:
        """Main audio processing loop with wake word detection."""
        try:
            while True:
                audio_data = self.stream_input.read(self.config.chunk_size_ow, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)

                # AIDEV-NOTE: Wake word detection using OpenWakeWord - optimized chunk size for performance
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
                    await self.processing_spoken_commands()

                # AIDEV-NOTE: Small yield to allow other async tasks to run
                await asyncio.sleep(0.001)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            raise

    async def _queue_processing_loop(self) -> None:
        """Background processing of TTS/STT queues."""
        try:
            while True:
                await self.process_output_queue()
                await self._check_stt_results()

                # AIDEV-NOTE: Small delay to prevent busy-waiting
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            self.logger.info("Queue processing interrupted")
            raise

    def start(self) -> None:
        """Start the satellite with MQTT client and audio processing."""
        self._start_mqtt_loop()

        # AIDEV-NOTE: Start API processing thread for non-blocking TTS/STT
        api_thread = threading.Thread(target=self._api_processor, daemon=True)
        api_thread.start()

        # AIDEV-NOTE: Run async event loop with concurrent tasks
        async def run_concurrent_tasks():
            # Create concurrent tasks for different processing loops
            audio_task = asyncio.create_task(self._audio_processing_loop())
            queue_task = asyncio.create_task(self._queue_processing_loop())

            # Run tasks concurrently until one completes (or raises exception)
            try:
                await asyncio.gather(audio_task, queue_task)
            except KeyboardInterrupt:
                self.logger.info("Shutting down concurrent tasks")
                audio_task.cancel()
                queue_task.cancel()
                raise

        try:
            asyncio.run(run_concurrent_tasks())
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
