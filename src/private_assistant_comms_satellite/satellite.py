from __future__ import annotations

import asyncio
import enum
import os
import queue
import threading
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import numpy as np

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
from private_assistant_comms_satellite.audio_processing import ParametricEQ, SimpleAGC
from private_assistant_comms_satellite.utils import (
    config,
    ground_station_client,
)


class SatelliteState(enum.Enum):
    """Satellite operation states."""

    LISTENING = "listening"  # Analyzing audio for wakeword
    RECORDING = "recording"  # Recording user command after wakeword
    WAITING = "waiting"  # Waiting for ground station response
    SPEAKING = "speaking"  # Playing TTS audio


class Satellite:
    """Main satellite class for voice interaction with ground station integration."""

    def __init__(  # noqa: PLR0913
        self,
        config: config.Config,
        start_listening_sound: bytes | np.ndarray,
        stop_listening_sound: bytes | np.ndarray,
        disconnection_sound: np.ndarray,
        wakeword_model: openwakeword.Model,
        logger: logging.Logger,
    ):
        # Assign configuration
        self.config = config
        # Assign preloaded sound data (bytes or arrays)
        self.start_listening_sound: bytes | np.ndarray = start_listening_sound
        self.stop_listening_sound: bytes | np.ndarray = stop_listening_sound
        self.vad_model = silero_vad.SileroVad(threshold=config.vad_threshold, trigger_level=config.vad_trigger)
        self.logger = logger
        # Assign wakeword model
        self.wakeword_model: openwakeword.Model = wakeword_model

        # AIDEV-NOTE: Ground station client replaces MQTT functionality
        self.ground_station = ground_station_client.GroundStationClient(config)

        # AIDEV-NOTE: Simple state machine
        self._state = SatelliteState.LISTENING
        self._state_lock = threading.RLock()

        # AIDEV-NOTE: Audio recording buffer
        max_recording_samples = self.config.max_command_input_seconds * self.config.samplerate
        self._audio_buffer = np.zeros(max_recording_samples, dtype=np.float32)
        self._buffer_position = 0

        # AIDEV-NOTE: Audio queues for callback-based streaming
        # AIDEV-NOTE: Small queue size for real-time processing - prevents delays in wakeword detection
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=10)
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
        self._disconnection_sound_array = self._to_audio_array(disconnection_sound)

        # Initialize parametric EQ (Phase 1)
        self.voice_eq: ParametricEQ | None = None
        if config.audio_processing.enable_voice_eq:
            self.voice_eq = ParametricEQ(
                sample_rate=config.samplerate,
                presence_boost_db=config.audio_processing.eq_presence_boost_db,
                presence_freq_hz=config.audio_processing.eq_presence_freq_hz,
                presence_q=config.audio_processing.eq_presence_q,
            )
            logger.info(
                "Voice EQ enabled: presence +%sdB @ %sHz",
                config.audio_processing.eq_presence_boost_db,
                config.audio_processing.eq_presence_freq_hz,
            )

        # Initialize AGC (Phase 2 - disabled by default)
        self.agc: SimpleAGC | None = None
        if config.audio_processing.enable_agc:
            self.agc = SimpleAGC(
                target_rms=config.audio_processing.agc_target_rms,
                smoothing=config.audio_processing.agc_smoothing,
            )
            logger.info("AGC enabled: target RMS %s", config.audio_processing.agc_target_rms)

        # Debug audio recording (set DEBUG_AUDIO=1 environment variable)
        self.debug_recorder = None
        self._debug_original_buffer: list[np.ndarray] = []
        self._debug_filtered_buffer: list[np.ndarray] = []
        if os.getenv("DEBUG_AUDIO") == "1":
            from private_assistant_comms_satellite.utils.debug_audio import AudioRecorder  # noqa: PLC0415

            self.debug_recorder = AudioRecorder()
            logger.info("Audio debug recording enabled")

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
            # Extract mono audio channel (float32, range -1.0 to 1.0)
            audio_float32 = indata[:, 0].copy()

            # AIDEV-NOTE: Two-stage audio enhancement pipeline for wake word detection
            # Stage 1: Parametric EQ (boost presence for clarity)
            if self.voice_eq is not None:
                audio_float32 = self.voice_eq.process(audio_float32)

            # Stage 2: AGC (normalize volume)
            if self.agc is not None:
                audio_float32 = self.agc.process(audio_float32)

            # Debug: accumulate audio during recording for comparison
            if (
                self.debug_recorder
                and self._state == SatelliteState.RECORDING
                and (self.voice_eq is not None or self.agc is not None)
            ):
                # Save original vs processed
                self._debug_original_buffer.append(indata[:, 0].copy())
                self._debug_filtered_buffer.append(audio_float32.copy())

            # Convert float32 to int16 for OpenWakeWord (range -32768 to 32767)
            audio_int16 = (audio_float32 * 32768.0).astype(np.int16)

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

    async def _process_ground_station_responses(self) -> None:
        """Process responses from ground station."""
        while self._running:
            try:
                # AIDEV-NOTE: Automatic reconnection handles disconnections, just wait for responses
                # Check for responses from ground station
                response = await self.ground_station.get_response(timeout=1.0)
                if response is None:
                    continue

                if response["type"] == "audio":
                    # TTS audio response
                    audio_bytes = response["data"]
                    self.logger.info("Received TTS audio from ground station (%d bytes)", len(audio_bytes))

                    self._set_state(SatelliteState.SPEAKING)

                    # Convert audio bytes to sounddevice format and queue for playback
                    tts_audio_array = self._bytes_to_audio_array(audio_bytes)
                    self._queue_sound(tts_audio_array)

                    # Wait for playback to complete
                    estimated_duration = len(tts_audio_array) / self._samplerate
                    max_wait_time = estimated_duration + 2.0  # Maximum wait with buffer
                    start_wait = time.time()

                    while self._is_sound_playing() and (time.time() - start_wait) < max_wait_time:
                        await asyncio.sleep(0.1)

                    self.logger.debug("TTS playback completed")
                    self._set_state(SatelliteState.LISTENING)

                elif response["type"] in ("text", "json"):
                    # Text or JSON response (alerts, errors, etc.)
                    self.logger.info("Received text response from ground station: %s", response["data"])

            except Exception as e:
                if self._running:  # Only log errors if still running
                    self.logger.error("Error processing ground station response: %s", e)
                await asyncio.sleep(0.1)

    def _reset_audio_buffer(self) -> None:
        """Reset the audio buffer for new recording."""
        self._buffer_position = 0

        # Reset all audio processors
        if self.voice_eq is not None:
            self.voice_eq.reset()
        if self.agc is not None:
            self.agc.reset()

        # Clear debug buffers for new recording
        self._debug_original_buffer.clear()
        self._debug_filtered_buffer.clear()

    def _save_debug_audio(self) -> None:
        """Save accumulated debug audio for comparison analysis."""
        if not self.debug_recorder or not self._debug_original_buffer or not self._debug_filtered_buffer:
            return

        # Concatenate all chunks into complete recordings
        original_audio = np.concatenate(self._debug_original_buffer)
        filtered_audio = np.concatenate(self._debug_filtered_buffer)

        # Save comparison
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.debug_recorder.save_comparison(original_audio, filtered_audio, self._samplerate, f"recording_{timestamp}")
        duration_sec = len(original_audio) / self._samplerate
        self.logger.info("Saved debug audio comparison: %d samples (%.2fs)", len(original_audio), duration_sec)

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

    async def _record_command(self) -> None:  # noqa: PLR0912, PLR0915
        """Record voice command after wake word detection."""
        self._set_state(SatelliteState.RECORDING)
        self._reset_audio_buffer()

        try:
            # AIDEV-NOTE: Automatic reconnection handles connection state, check if connected
            if not self.ground_station.is_connected:
                self.logger.info("Ground station not connected, waiting for automatic reconnection...")
                # Wait briefly for automatic reconnection
                max_wait = 5.0  # Wait up to 5 seconds
                wait_start = time.time()
                while not self.ground_station.is_connected and (time.time() - wait_start) < max_wait:
                    await asyncio.sleep(0.5)

                if not self.ground_station.is_connected:
                    self.logger.warning("Ground station still not connected after waiting")
                    # Play disconnection warning sound to alert user
                    self._queue_sound(self._disconnection_sound_array)
                    # Save any debug audio captured before disconnection
                    self._save_debug_audio()
                    # Return to listening state
                    self._set_state(SatelliteState.LISTENING)
                    return

            # Send START_COMMAND to ground station
            await self.ground_station.send_start_command()

            silence_packages = 0
            max_frames = self.config.max_command_input_seconds * self.config.samplerate
            max_silent_packages = self.config.samplerate / self.config.chunk_size * self.config.max_length_speech_pause
            has_audio_data = False
            active_listening = True
            total_packages = 0  # AIDEV-NOTE: Fallback counter to prevent infinite recording

            while active_listening:
                try:
                    # Get audio data from callback queue (non-blocking)
                    audio_int16 = self._audio_queue.get_nowait()
                    # Convert int16 to float32 for VAD model (which expects float32 arrays)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    total_packages += 1  # AIDEV-NOTE: Track total packages processed

                    if self.vad_model(audio_float32):
                        # Use already converted float32 audio for buffer storage
                        audio_chunk = audio_float32

                        silence_packages = 0
                        if not self._append_to_audio_buffer(audio_chunk):
                            active_listening = False
                            self.logger.warning("Audio buffer full, stopping recording")
                        else:
                            has_audio_data = True
                            # Send audio chunk to ground station (convert back to bytes)
                            audio_bytes = audio_int16.tobytes()
                            await self.ground_station.send_audio_chunk(audio_bytes)
                        self.logger.debug("Received voice...")
                    else:
                        if has_audio_data:
                            # Use already converted float32 audio for buffer storage
                            audio_chunk = audio_float32

                            if not self._append_to_audio_buffer(audio_chunk):
                                active_listening = False
                                self.logger.warning("Audio buffer full, stopping recording")
                            else:
                                silence_packages += 1
                                # Send silence chunk to ground station too (convert back to bytes)
                                audio_bytes = audio_int16.tobytes()
                                await self.ground_station.send_audio_chunk(audio_bytes)
                        self.logger.debug("No voice...")

                    # AIDEV-NOTE: Enhanced termination conditions to prevent infinite recording
                    buffer_full = self._buffer_position > max_frames
                    silence_exceeded = silence_packages >= max_silent_packages
                    max_packages = self.config.max_command_input_seconds * (
                        self.config.samplerate / self.config.chunk_size_ow
                    )
                    timeout_exceeded = total_packages >= max_packages

                    # Stop recording if:
                    # 1. We had audio data and (buffer full or silence exceeded), OR
                    # 2. Maximum recording time exceeded (fallback)
                    if (has_audio_data and (buffer_full or silence_exceeded)) or timeout_exceeded:
                        active_listening = False
                        if timeout_exceeded and not has_audio_data:
                            self.logger.warning("Recording timeout - no speech detected, stopping")
                        else:
                            self.logger.debug("Stopping listening, playing stop sound...")
                        self._queue_sound(self._stop_sound_array)

                        # Save debug audio comparison if enabled
                        self._save_debug_audio()

                        # Send END_COMMAND to ground station
                        await self.ground_station.send_end_command()

                        # Return to listening immediately - ground station can send responses anytime
                        self._set_state(SatelliteState.LISTENING)

                except queue.Empty:
                    # No audio data available, yield control
                    await asyncio.sleep(0.001)

        except Exception as e:
            self.logger.error("Error in recording: %s", e)
            # Save debug audio if any was captured before error
            self._save_debug_audio()
            # Send cancel command in case of error
            with suppress(Exception):
                await self.ground_station.send_cancel_command()
        finally:
            # Ensure we return to listening if still in recording state (error case)
            current_state = self._get_state()
            if current_state == SatelliteState.RECORDING:
                self._set_state(SatelliteState.LISTENING)

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
        """Main processing loop - handles wake word detection and ground station communication."""
        try:
            # Start audio processing task
            audio_task = asyncio.create_task(self._process_audio_queue())
            # Start ground station response handler
            response_task = asyncio.create_task(self._process_ground_station_responses())

            # Wait for either task to complete or fail
            done, pending = await asyncio.wait([audio_task, response_task], return_when=asyncio.FIRST_EXCEPTION)

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

            # Re-raise any exceptions
            for task in done:
                task.result()

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            raise

    def start(self) -> None:
        """Start the satellite with ground station integration."""
        self.logger.info("Starting satellite with ground station integration...")

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

            # AIDEV-NOTE: Run ground station client with automatic reconnection and main processing loop
            async def run_with_ground_station():
                # Start automatic reconnection in background
                reconnection_task = asyncio.create_task(self.ground_station.start_with_reconnection())

                try:
                    # Run main loop until stopped or error
                    await self._main_loop()
                except Exception as e:
                    self.logger.error("Main loop error: %s", e)
                    raise
                finally:
                    # Stop automatic reconnection
                    await self.ground_station.stop()
                    # Wait for reconnection task to complete
                    with suppress(asyncio.CancelledError):
                        await reconnection_task

            asyncio.run(run_with_ground_station())

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            raise
        except Exception as e:
            self.logger.error("Error starting satellite: %s", e)
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources including ground station connection and audio streams."""
        self.logger.info("Cleaning up satellite...")
        self._running = False

        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()

        if self._output_stream:
            self._output_stream.stop()
            self._output_stream.close()
