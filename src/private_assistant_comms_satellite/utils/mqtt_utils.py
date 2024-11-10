import logging
import queue
from collections.abc import Callable

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


def get_mqtt_event_functions(
    topic_to_queue: dict[str, queue.Queue[str]],
) -> tuple[Callable, Callable]:
    def on_message(client: mqtt.Client, user_data, message: mqtt.MQTTMessage):
        topic_queue = topic_to_queue.get(message.topic)
        logger.debug("Received message: %s", message)
        if topic_queue is None:
            logger.warning("%s seems to have no queue. Discarding message.", message.topic)
        else:
            topic_queue.put(message.payload.decode("utf-8"))

    def on_connect(
        client: mqtt.Client,
        user_data,
        flags: mqtt.ConnectFlags,
        reason_code,
        properties,
    ):
        logger.info(f"Connected to MQTT server with result code {reason_code}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        for sub in topic_to_queue:
            client.subscribe(sub, options=mqtt.SubscribeOptions(qos=1))

    return on_connect, on_message
