import asyncio
import queue
import ssl

import aiomqtt
from private_assistant_commons import messages, skill_logger

logger = skill_logger.SkillLogger.get_logger("Private Assistant Comms Satellite")


class AsyncMQTTClient:
    """Async MQTT client wrapper for aiomqtt integration."""

    def __init__(  # noqa: PLR0913
        self,
        hostname: str,
        port: int,
        client_id: str,
        topic_to_queue: dict[str, asyncio.Queue[messages.Response]],
        use_websockets: bool = False,
        websocket_path: str = "/mqtt",
        use_ssl: bool = False,
    ):
        self.hostname = hostname
        self.port = port
        self.client_id = client_id
        self.topic_to_queue: dict[str, asyncio.Queue[messages.Response]] = topic_to_queue
        self.use_websockets = use_websockets
        self.websocket_path = websocket_path
        self.use_ssl = use_ssl
        self._client: aiomqtt.Client | None = None
        self._running = False

    async def _handle_client_connection(self, client: aiomqtt.Client) -> None:
        """Handle MQTT client connection, subscription and message processing."""
        self._client = client
        logger.info("Connected to MQTT server")

        # Subscribe to all topics
        for topic in self.topic_to_queue:
            await client.subscribe(topic, qos=1)
            logger.debug("Subscribed to topic: %s", topic)

        # Handle messages indefinitely
        async for message in client.messages:
            topic_queue = self.topic_to_queue.get(message.topic.value)
            logger.debug("Received message on topic %s: %s", message.topic, message.payload)

            if topic_queue is None:
                logger.warning("%s seems to have no queue. Discarding message.", message.topic.value)
            else:
                try:
                    # AIDEV-NOTE: Payload parsing handles various aiomqtt message formats for robustness
                    if isinstance(message.payload, bytes | bytearray):
                        payload_str = message.payload.decode("utf-8")
                    elif isinstance(message.payload, str):
                        payload_str = message.payload
                    else:
                        payload_str = str(message.payload)
                    topic_queue.put_nowait(messages.Response.model_validate_json(payload_str))
                except queue.Full:
                    logger.warning("Queue for topic %s is full, discarding message", message.topic.value)

    async def start(self) -> None:
        """Start the MQTT client and message handling."""
        if self._running:
            return

        self._running = True
        logger.info("Starting MQTT client connection to %s:%s", self.hostname, self.port)
        logger.info("WebSocket mode: %s, SSL mode: %s", self.use_websockets, self.use_ssl)
        if self.use_websockets:
            logger.info("WebSocket path: %s", self.websocket_path)

        try:
            # Configure SSL context if needed
            tls_context = None
            if self.use_ssl:
                tls_context = ssl.create_default_context()
                logger.info("Using SSL/TLS encryption")

            # Configure transport based on WebSocket setting
            if self.use_websockets:
                if self.use_ssl:
                    logger.info("Using secure WebSocket (WSS) transport with path: %s", self.websocket_path)
                else:
                    logger.info("Using WebSocket (WS) transport with path: %s", self.websocket_path)

                logger.debug("Creating WebSocket client with SSL context: %s", tls_context is not None)
                async with aiomqtt.Client(
                    hostname=self.hostname,
                    port=self.port,
                    identifier=self.client_id,
                    transport="websockets",
                    websocket_path=self.websocket_path,
                    tls_context=tls_context,
                    timeout=30,  # Add explicit timeout
                ) as client:
                    await self._handle_client_connection(client)
            else:
                if self.use_ssl:
                    logger.info("Using MQTT over SSL/TLS")

                async with aiomqtt.Client(
                    hostname=self.hostname,
                    port=self.port,
                    identifier=self.client_id,
                    tls_context=tls_context,
                ) as client:
                    await self._handle_client_connection(client)
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
