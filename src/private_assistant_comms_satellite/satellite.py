from __future__ import annotations

import asyncio
import enum
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from private_assistant_commons import messages

try:
    import sounddevice as sd
except OSError as e:
    if "PortAudio library not found" in str(e):
        # Expected in environments without audio hardware
        sd = None
    else:
        raise

if TYPE_CHECKING:
    import logging

    import openwakeword

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
        start_listening_sound: bytes | np.ndarray,
        stop_listening_sound: bytes | np.ndarray,
        wakeword_model: openwakeword.Model,
        mqtt_client: mqtt_utils.AsyncMQTTClient,
        logger: logging.Logger,
    ):
        # Assign configuration
        self.config = config
        self.output_queue = output_queue
        # Assign preloaded sound data (bytes or arrays)
        self.start_listening_sound: bytes | np.ndarray = start_listening_sound
        self.stop_listening_sound: bytes | np.ndarray = stop_listening_sound
        self.vad_model = silero_vad.SileroVad(threshold=config.vad_threshold, trigger_level=config.vad_trigger)
        self.logger = logger
        # Assign wakeword model
        self.wakeword_model: openwakeword.Model = wakeword_model
        self.mqtt_client = mqtt_client
        self._mqtt_thread: threading.Thread | None = None
        self._mqtt_loop: asyncio.AbstractEventLoop | None = None

        # AIDEV-NOTE: Simple state machine
        self._state = SatelliteState.LISTENING
        self._state_lock = threading.RLock()

        # AIDEV-NOTE: Audio recording buffer
        max_recording_samples = self.config.max_command_input_seconds * self.config.samplerate
        self._audio_buffer = np.zeros(max_recording_samples, dtype=np.float32)
        self._buffer_position = 0

        # AIDEV-NOTE: Timing tracking
        self._stop_sound_time = 0.0

        # AIDEV-NOTE: Audio queues for callback-based streaming
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._output_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._output_position = 0
        self._running = False
        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None

        # AIDEV-NOTE: Sounddevice uses float32 internally but we'll work with int16 for OpenWakeWord compatibility
        self._audio_dtype = np.int16
        self._samplerate = config.samplerate
        self._chunk_size_ow = config.chunk_size_ow

        # AIDEV-NOTE: Convert notification sounds to numpy arrays for sounddevice
        self._start_sound_array = self._to_audio_array(start_listening_sound)
        self._stop_sound_array = self._to_audio_array(stop_listening_sound)

    def _to_audio_array(self, audio_data: bytes | np.ndarray) -> np.ndarray:
        """Convert audio data (bytes or array) to numpy array suitable for sounddevice playback."""
        if isinstance(audio_data, bytes):
            # Convert int16 bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Sounddevice expects float32 in range [-1.0, 1.0] for optimal performance
            return audio_array.astype(np.float32) / 32768.0
        if isinstance(audio_data, np.ndarray):
            # Already a numpy array (should be float32 from sound generation)
            if audio_data.dtype == np.float32:
                return audio_data.copy()
            # Convert to float32 if needed
            return audio_data.astype(np.float32)
        raise TypeError(f"Unsupported audio data type: {type(audio_data)}")

    def _bytes_to_audio_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array suitable for sounddevice playback."""
        # Convert int16 bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        # Sounddevice expects float32 in range [-1.0, 1.0] for optimal performance
        return audio_array.astype(np.float32) / 32768.0

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:  # noqa: ARG002
        """Audio input callback - processes incoming audio for wakeword detection."""
        if status:
            self.logger.warning("Audio callback status: %s", status)

        try:
            # Convert float32 back to int16 for OpenWakeWord
            audio_int16 = (indata[:, 0] * 32768.0).astype(np.int16)

            # AIDEV-NOTE: Put audio data in queue for processing thread
            self._audio_queue.put_nowait(audio_int16)
        except queue.Full:
            self.logger.warning("Audio queue full, dropping frame")

    def _output_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:  # noqa: ARG002
        """Audio output callback - plays notification sounds."""
        if status:
            self.logger.warning("Audio output callback status: %s", status)

        # Fill output with silence by default
        outdata.fill(0)

        # Check if we have audio data to play
        if len(self._output_buffer) > self._output_position:
            # Calculate how many samples we can copy
            remaining_samples = len(self._output_buffer) - self._output_position
            samples_to_copy = min(frames, remaining_samples)

            # Copy audio data to output
            start_pos = self._output_position
            end_pos = self._output_position + samples_to_copy
            outdata[:samples_to_copy, 0] = self._output_buffer[start_pos:end_pos]
            self._output_position += samples_to_copy

            # Reset buffer when finished
            if self._output_position >= len(self._output_buffer):
                self._output_buffer = np.array([], dtype=np.float32)
                self._output_position = 0

    def _queue_sound(self, sound_array: np.ndarray) -> None:
        """Queue a sound for playback via output stream."""
        self._output_buffer = sound_array.copy()
        self._output_position = 0
        self.logger.debug("Queued sound for playback (duration: %.2f seconds)", len(sound_array) / self._samplerate)

    def _is_sound_playing(self) -> bool:
        """Check if sound is currently playing."""
        return len(self._output_buffer) > 0 and self._output_position < len(self._output_buffer)

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
            self.logger.info("Received TTS message: '%s'", response.text)

            # AIDEV-NOTE: Set state to waiting, then speaking
            self._set_state(SatelliteState.WAITING)

            try:
                # Get TTS audio synchronously
                tts_start_time = time.time()
                audio_bytes = await speech_recognition_tools.send_text_to_tts_api(response.text, self.config)
                tts_processing_time = time.time() - tts_start_time

                self.logger.debug("TTS processing completed in %.3f seconds", tts_processing_time)

                if audio_bytes:
                    self.logger.info("TTS audio received, starting playback")
                    self._set_state(SatelliteState.SPEAKING)
                    playback_start_time = time.time()
                    # Calculate total processing time from stop sound to TTS playback start
                    if self._stop_sound_time > 0:
                        total_processing_time = playback_start_time - self._stop_sound_time
                        self.logger.debug(
                            "Total processing time (STT + LLM + TTS): %.3f seconds", total_processing_time
                        )

                    self.logger.debug("Starting TTS playback at: %.3f", playback_start_time)
                    # Convert audio bytes to sounddevice format and queue for playback
                    tts_audio_array = self._bytes_to_audio_array(audio_bytes)
                    self._queue_sound(tts_audio_array)

                    # Wait for playback to actually complete by polling
                    estimated_duration = len(tts_audio_array) / self._samplerate
                    self.logger.debug(
                        "Waiting for TTS audio playback to complete (estimated: %.2f seconds)", estimated_duration
                    )

                    max_wait_time = estimated_duration + 5.0  # Maximum wait with buffer
                    start_wait = time.time()

                    while self._is_sound_playing() and (time.time() - start_wait) < max_wait_time:
                        await asyncio.sleep(0.1)  # Check every 100ms

                    playback_duration = time.time() - playback_start_time
                    self.logger.debug("TTS playback completed after %.3f seconds", playback_duration)
                    self.logger.debug(
                        "Finished playing TTS audio (TTS processing: %.3f seconds, playback duration: %.3f seconds)",
                        tts_processing_time,
                        playback_duration,
                    )
                else:
                    self.logger.warning("No TTS audio received from API")

            except Exception as e:
                self.logger.error("Error in TTS processing: %s", e, exc_info=True)
            finally:
                # Always return to listening state, even if there was an error
                self.logger.info("Returning to listening state")
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
        recording_start_time = time.time()

        try:
            silence_packages = 0
            max_frames = self.config.max_command_input_seconds * self.config.samplerate
            max_silent_packages = self.config.samplerate / self.config.chunk_size * self.config.max_length_speech_pause
            has_audio_data = False
            active_listening = True

            while active_listening:
                try:
                    # Get audio data from callback queue (non-blocking)
                    audio_int16 = self._audio_queue.get_nowait()
                    # Convert to bytes for VAD model (which expects bytes)
                    audio_bytes = audio_int16.tobytes()

                    if self.vad_model(audio_bytes) > self.config.vad_threshold:
                        # Convert int16 to float32 for buffer storage
                        audio_chunk = audio_int16.astype(np.float32) / 32768.0

                        silence_packages = 0
                        if not self._append_to_audio_buffer(audio_chunk):
                            active_listening = False
                            self.logger.warning("Audio buffer full, stopping recording")
                        else:
                            has_audio_data = True
                        self.logger.debug("Received voice...")
                    else:
                        if has_audio_data:
                            # Convert int16 to float32 for buffer storage
                            audio_chunk = audio_int16.astype(np.float32) / 32768.0

                            if not self._append_to_audio_buffer(audio_chunk):
                                active_listening = False
                                self.logger.warning("Audio buffer full, stopping recording")
                            else:
                                silence_packages += 1
                        self.logger.debug("No voice...")

                    buffer_full = self._buffer_position > max_frames
                    silence_exceeded = silence_packages >= max_silent_packages
                    if has_audio_data and (buffer_full or silence_exceeded):
                        active_listening = False
                        recording_duration = time.time() - recording_start_time
                        self.logger.debug("Stopping listening, playing stop sound...")
                        self._queue_sound(self._stop_sound_array)
                        self.logger.debug("Recording duration: %.3f seconds", recording_duration)

                        # Process STT
                        audio_frames = self._get_recorded_audio()
                        if len(audio_frames) > 0:
                            # Store stop sound time for total processing calculation
                            self._stop_sound_time = time.time()
                            await self._process_stt(audio_frames)

                except queue.Empty:
                    # No audio data available, yield control
                    await asyncio.sleep(0.001)

        finally:
            self._set_state(SatelliteState.LISTENING)

    async def _process_stt(self, audio_frames: np.ndarray) -> None:
        """Process speech-to-text and publish to MQTT."""
        self._set_state(SatelliteState.WAITING)
        stt_start_time = time.time()
        self.logger.debug("Processing STT request")

        try:
            response = await speech_recognition_tools.send_audio_to_stt_api(audio_frames, config_obj=self.config)
            stt_processing_time = time.time() - stt_start_time

            if response is not None:
                self.logger.debug(
                    "STT result: '%s' (processing time: %.3f seconds)", response.text, stt_processing_time
                )

                message = messages.ClientRequest(
                    id=uuid.uuid4(),
                    text=response.text,
                    room=self.config.room,
                    output_topic=self.config.output_topic,
                ).model_dump_json()

                # AIDEV-NOTE: Schedule MQTT publish on the MQTT thread's event loop
                mqtt_publish_start = time.time()
                if self._mqtt_loop and not self._mqtt_loop.is_closed():
                    future = asyncio.run_coroutine_threadsafe(
                        self.mqtt_client.publish(self.config.input_topic, message, qos=1), self._mqtt_loop
                    )
                    # Wait for the publish to complete
                    future.result(timeout=5)
                    mqtt_publish_time = time.time() - mqtt_publish_start
                    self.logger.debug("Published result text to MQTT (publish time: %.3f seconds).", mqtt_publish_time)
                else:
                    self.logger.error("MQTT loop not available, cannot publish STT result")
            else:
                self.logger.warning("STT processing failed for recorded audio")
        except Exception as e:
            self.logger.error("STT processing failed: %s", e)
        finally:
            self._set_state(SatelliteState.LISTENING)

    def _start_mqtt_loop(self) -> None:
        """Start the MQTT client in a separate thread."""

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
        time.sleep(1)  # Give MQTT client time to start

    async def _process_audio_queue(self) -> None:
        """Process audio queue for wakeword detection and recording."""
        while self._running:
            try:
                current_state = self._get_state()

                # Get audio data from queue (non-blocking) - always drain the queue to prevent buildup
                try:
                    audio_chunk = self._audio_queue.get_nowait()

                    # Only process audio for wake word in LISTENING state
                    if current_state == SatelliteState.LISTENING:
                        # AIDEV-NOTE: Process wakeword detection with OpenWakeWord
                        prediction = self.wakeword_model.predict(
                            audio_chunk,
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
                            self.logger.debug("Wakeword detected, playing start listening sound.")
                            self._queue_sound(self._start_sound_array)
                            await self._record_command()
                    else:
                        # In other states, just drain the queue to prevent buildup
                        self.logger.debug("Draining audio queue (state: %s)", current_state.value)

                except queue.Empty:
                    # No audio data available, yield control
                    await asyncio.sleep(0.001)

            except Exception as e:
                self.logger.error("Error processing audio: %s", e)
                await asyncio.sleep(0.01)

    async def _main_loop(self) -> None:
        """Main processing loop - handles wake word detection and state transitions."""
        try:
            # Start audio processing task
            audio_task = asyncio.create_task(self._process_audio_queue())

            while True:
                # Always check for TTS messages (low latency MQTT processing)
                await self._process_output_queue()

                # Small yield to prevent busy-waiting
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            audio_task.cancel()
            raise

    def start(self) -> None:
        """Start the satellite with simple state machine."""
        self.logger.info("Starting sounddevice satellite...")

        if sd is None:
            raise ImportError(
                "sounddevice is required for audio processing but PortAudio library not found. "
                "Please install system audio dependencies (see README.md)"
            )

        try:
            self._running = True

            # AIDEV-NOTE: Initialize input stream with callback-based approach for device consistency
            has_input_device = hasattr(self.config, "input_device_index")
            input_device = None
            if has_input_device and self.config.input_device_index is not None:
                input_device = self.config.input_device_index
            self._input_stream = sd.InputStream(
                samplerate=self._samplerate,
                channels=1,
                dtype=np.float32,  # Sounddevice native format
                blocksize=self._chunk_size_ow,  # Fixed chunk size for OpenWakeWord
                callback=self._audio_callback,
                device=input_device,  # Use configured input device
            )

            # AIDEV-NOTE: Initialize output stream for notification sounds using same device
            has_output_device = hasattr(self.config, "output_device_index")
            output_device = None
            if has_output_device and self.config.output_device_index is not None:
                output_device = self.config.output_device_index
            if output_device is None and input_device is not None:
                # If no output device specified but input is, try to use same device
                output_device = input_device

            self._output_stream = sd.OutputStream(
                samplerate=self._samplerate,
                channels=1,
                dtype=np.float32,  # Sounddevice native format
                callback=self._output_callback,
                device=output_device,  # Use same device as input for consistency
            )

            self.logger.info("Audio stream configuration:")
            self.logger.info("  Sample rate: %d Hz", self._samplerate)
            self.logger.info("  Chunk size: %d samples", self._chunk_size_ow)
            self.logger.info("  Input device: %s", self._input_stream.device)
            self.logger.info("  Output device: %s", self._output_stream.device)

            # Start both streams
            self._input_stream.start()
            self._output_stream.start()

            # Start MQTT client in separate thread
            self._start_mqtt_loop()

            # Run main processing loop
            asyncio.run(self._main_loop())

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            raise
        except Exception as e:
            self.logger.error("Error starting satellite: %s", e)
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources including MQTT client and audio streams."""
        self.logger.info("Cleaning up sounddevice satellite...")
        self._running = False

        if self._mqtt_thread and self._mqtt_thread.is_alive():
            self._mqtt_thread.join(timeout=2.0)

        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()

        if self._output_stream:
            self._output_stream.stop()
            self._output_stream.close()
