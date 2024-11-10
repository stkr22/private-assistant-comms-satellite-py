import logging
import queue
import uuid

import numpy as np
import openwakeword
import paho.mqtt.client as mqtt
import pyaudio
from private_assistant_commons import messages

from private_assistant_comms_satellite import silero_vad
from private_assistant_comms_satellite.utils import (
    config,
    speech_recognition_tools,
)


class Satellite:
    def __init__(
        self,
        config: config.Config,
        output_queue: queue.Queue[str],
        start_listening_sound: bytes,
        stop_listening_sound: bytes,
        wakeword_model: openwakeword.Model,
        mqtt_client: mqtt.Client,
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

        # Initialize PyAudio
        self.p: pyaudio.PyAudio = pyaudio.PyAudio()
        self.stream_input: pyaudio.Stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.samplerate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            input_device_index=self.config.input_device_index,
        )
        self.stream_output: pyaudio.Stream = self.p.open(
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
            audio_np = speech_recognition_tools.send_text_to_tts_api(output_text, self.config)
            if audio_np is not None:
                self.stream_output.write(audio_np.tobytes())
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
            # speech_prob, data = self.format_audio_and_speech_prob(raw_audio_data)
            if self.vad_model(audio_bytes) > self.config.vad_threshold:
                raw_audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                data = speech_recognition_tools.int2float(raw_audio_data)
                silence_packages = 0
                if audio_frames is None:
                    audio_frames = data
                else:
                    audio_frames = np.concatenate((audio_frames, data), axis=0)
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
                response = speech_recognition_tools.send_audio_to_stt_api(audio_frames, config_obj=self.config)
                self.logger.info("Received result...%s", response)
                if response is not None:
                    self.mqtt_client.publish(
                        self.config.input_topic,
                        messages.ClientRequest(
                            id=uuid.uuid4(),
                            text=response["text"],
                            room=self.config.room,
                            output_topic=self.config.output_topic,
                        ).model_dump_json(),
                        qos=1,
                    )
                    self.logger.info("Published result text to MQTT.")

    def start(self) -> None:
        self.mqtt_client.loop_start()
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

    def cleanup(self) -> None:
        self.mqtt_client.loop_stop()
        self.stream_input.stop_stream()
        self.stream_output.stop_stream()
        self.stream_input.close()
        self.stream_output.close()
        self.p.terminate()
