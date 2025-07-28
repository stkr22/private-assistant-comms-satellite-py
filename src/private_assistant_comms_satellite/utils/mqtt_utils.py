import logging
import queue

import aiomqtt

logger = logging.getLogger(__name__)


class AsyncMQTTClient:
    """Async MQTT client wrapper for aiomqtt integration."""
    def __init__(self, hostname: str, port: int, client_id: str, topic_to_queue: dict[str, queue.Queue[str]]):
        self.hostname = hostname
        self.port = port
        self.client_id = client_id
        self.topic_to_queue = topic_to_queue
        self._client: aiomqtt.Client | None = None
        self._running = False

    async def start(self) -> None:
        """Start the MQTT client and message handling."""
        if self._running:
            return

        self._running = True
        logger.info("Starting MQTT client connection to %s:%s", self.hostname, self.port)

        try:
            async with aiomqtt.Client(
                hostname=self.hostname,
                port=self.port,
                identifier=self.client_id,
            ) as client:
                self._client = client
                logger.info("Connected to MQTT server")

                # Subscribe to all topics
                for topic in self.topic_to_queue:
                    await client.subscribe(topic, qos=1)
                    logger.info("Subscribed to topic: %s", topic)

                # Handle messages indefinitely
                async for message in client.messages:
                    topic_queue = self.topic_to_queue.get(message.topic.value)
                    logger.debug("Received message on topic %s: %s", message.topic, message.payload)

                    if topic_queue is None:
                        logger.warning("%s seems to have no queue. Discarding message.", message.topic.value)
                    else:
                        try:
                            # AIDEV-NOTE: Handle different payload types from aiomqtt
                            if isinstance(message.payload, bytes | bytearray):
                                payload_str = message.payload.decode("utf-8")
                            elif isinstance(message.payload, str):
                                payload_str = message.payload
                            else:
                                payload_str = str(message.payload)
                            topic_queue.put_nowait(payload_str)
                        except queue.Full:
                            logger.warning("Queue for topic %s is full, discarding message", message.topic.value)
        except Exception as e:
            logger.error("MQTT client error: %s", e)
        finally:
            self._running = False
            self._client = None

    async def publish(self, topic: str, payload: str, qos: int = 1) -> None:
        """Publish a message to the specified topic."""
        if not self._client:
            logger.error("MQTT client not connected, cannot publish")
            return

        try:
            await self._client.publish(topic, payload, qos=qos)
            logger.debug("Published message to %s: %s", topic, payload)
        except Exception as e:
            logger.error("Failed to publish message: %s", e)

    async def stop(self) -> None:
        """Stop the MQTT client."""
        self._running = False
        # AIDEV-NOTE: Client cleanup happens automatically in context manager
        logger.info("MQTT client stopped")
