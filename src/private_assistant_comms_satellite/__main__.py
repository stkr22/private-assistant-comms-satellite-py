import os
import pathlib
import queue
import wave

import openwakeword
import paho.mqtt.client as mqtt
from private_assistant_commons import skill_logger

from private_assistant_comms_satellite import satellite
from private_assistant_comms_satellite.utils import (
    config,
    mqtt_utils,
)


# Load WAV file into memory
def load_wav_file(file_path):
    wf = wave.open(file_path, "rb")
    audio_data = wf.readframes(wf.getnframes())
    wf.close()
    return audio_data


def main():
    config_obj = config.load_config(pathlib.Path(os.getenv("PRIVATE_ASSISTANT_API_CONFIG_PATH", "local_config.yaml")))
    wakeword_model = openwakeword.Model(
        wakeword_models=[config_obj.path_or_name_wakeword_model],
        enable_speex_noise_suppression=True,
        vad_threshold=config_obj.vad_threshold,
    )
    mqtt_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=config_obj.client_id,
        protocol=mqtt.MQTTv5,
    )
    topic_to_queue: dict[str, queue.Queue[str]] = {}
    output_queue: queue.Queue[str] = queue.Queue()
    output_topic = config_obj.output_topic
    topic_to_queue[output_topic] = output_queue
    topic_to_queue[config_obj.broadcast_topic] = output_queue
    mqtt_client.on_connect, mqtt_client.on_message = mqtt_utils.get_mqtt_event_functions(topic_to_queue)
    mqtt_client.connect(
        config_obj.mqtt_server_host,
        config_obj.mqtt_server_port,
        60,
    )

    logger = skill_logger.SkillLogger.get_logger("Private Assistant Comms Satellite")
    # Preload sounds
    start_listening_sound = load_wav_file(config_obj.start_listening_path)
    stop_listening_sound = load_wav_file(config_obj.stop_listening_path)
    satellite_handler = satellite.Satellite(
        config=config_obj,
        output_queue=output_queue,
        start_listening_sound=start_listening_sound,
        stop_listening_sound=stop_listening_sound,
        wakeword_model=wakeword_model,
        mqtt_client=mqtt_client,
        logger=logger,
    )

    try:
        satellite_handler.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, stopping...")
    finally:
        satellite_handler.cleanup()


if __name__ == "__main__":
    main()
