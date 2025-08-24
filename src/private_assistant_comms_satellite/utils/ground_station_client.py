import asyncio
import json
from typing import TYPE_CHECKING, Any

import websockets
from private_assistant_commons import skill_logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from private_assistant_comms_satellite.utils import config

logger = skill_logger.SkillLogger.get_logger("Private Assistant Comms Satellite")


class ClientConfig(BaseModel):
    """Configuration message schema matching ground station expectations."""

    samplerate: int
    input_channels: int
    output_channels: int
    chunk_size: int
    room: str
    output_topic: str = ""


class GroundStationClient:
    """WebSocket client for communicating with the ground station."""

    def __init__(self, config_obj: "config.Config"):
        self.config = config_obj
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self._connected = False
        self._response_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def connect(self) -> None:
        """Connect to the ground station WebSocket endpoint."""
        try:
            logger.info("Connecting to ground station: %s", self.config.ground_station_url)
            self.websocket = await websockets.connect(self.config.ground_station_url)
            self._connected = True
            logger.info("Connected to ground station successfully")

            # Send initial configuration using the same schema as ground station
            config_message = ClientConfig(
                samplerate=self.config.samplerate,
                input_channels=1,
                output_channels=1,
                chunk_size=self.config.chunk_size,
                room=self.config.room,
            )
            # Send as JSON using Pydantic's model_dump_json()
            await self.websocket.send(config_message.model_dump_json())  # type: ignore
            logger.debug("Sent configuration to ground station: %s", config_message.model_dump())

            # Start message handler
            self._message_task = asyncio.create_task(self._handle_messages())

        except Exception as e:
            logger.error("Failed to connect to ground station: %s", e)
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from the ground station."""
        self._connected = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from ground station")

    @property
    def is_connected(self) -> bool:
        """Check if connected to ground station."""
        return self._connected and self.websocket is not None

    async def send_start_command(self) -> None:
        """Send START_COMMAND signal to indicate wake word detected."""
        if not self.is_connected:
            logger.error("Not connected to ground station, cannot send start command")
            return

        try:
            await self.websocket.send("START_COMMAND")  # type: ignore
            logger.debug("Sent START_COMMAND to ground station")
        except Exception as e:
            logger.error("Failed to send start command: %s", e)

    async def send_audio_chunk(self, audio_data: bytes) -> None:
        """Send audio data chunk to ground station."""
        if not self.is_connected:
            logger.error("Not connected to ground station, cannot send audio")
            return

        try:
            await self.websocket.send(audio_data)  # type: ignore
            logger.debug("Sent audio chunk (%d bytes) to ground station", len(audio_data))
        except Exception as e:
            logger.error("Failed to send audio chunk: %s", e)

    async def send_end_command(self) -> None:
        """Send END_COMMAND signal to indicate end of audio input."""
        if not self.is_connected:
            logger.error("Not connected to ground station, cannot send end command")
            return

        try:
            await self.websocket.send("END_COMMAND")  # type: ignore
            logger.debug("Sent END_COMMAND to ground station")
        except Exception as e:
            logger.error("Failed to send end command: %s", e)

    async def send_cancel_command(self) -> None:
        """Send CANCEL_COMMAND signal to cancel current processing."""
        if not self.is_connected:
            logger.error("Not connected to ground station, cannot send cancel command")
            return

        try:
            await self.websocket.send("CANCEL_COMMAND")  # type: ignore
            logger.debug("Sent CANCEL_COMMAND to ground station")
        except Exception as e:
            logger.error("Failed to send cancel command: %s", e)

    async def get_response(self, timeout: float = 30.0) -> dict[str, Any] | None:
        """Get next response from ground station with timeout."""
        try:
            return await asyncio.wait_for(self._response_queue.get(), timeout=timeout)
        except TimeoutError:
            # Timeout is normal when no messages are available - don't log as warning
            return None

    async def _send_json(self, data: dict[str, Any]) -> None:
        """Send JSON message to ground station."""
        if not self.is_connected:
            return

        try:
            message = json.dumps(data)
            await self.websocket.send(message)  # type: ignore
        except Exception as e:
            logger.error("Failed to send JSON message: %s", e)

    async def _handle_messages(self) -> None:
        """Handle incoming messages from ground station."""
        if not self.websocket:
            return

        try:
            async for message in self.websocket:
                try:
                    if isinstance(message, bytes):
                        # Audio response (TTS)
                        response = {"type": "audio", "data": message}
                        await self._response_queue.put(response)
                        logger.debug("Received audio response (%d bytes)", len(message))
                    elif isinstance(message, str):
                        # Text/JSON response
                        try:
                            json_data = json.loads(message)
                            response = {"type": "json", "data": json_data}
                        except json.JSONDecodeError:
                            # Plain text response
                            response = {"type": "text", "data": message}

                        await self._response_queue.put(response)
                        logger.debug("Received text/json response: %s", message[:100])
                    else:
                        logger.warning("Unknown message type: %s", type(message))

                except Exception as e:
                    logger.error("Error processing message: %s", e)

        except websockets.exceptions.ConnectionClosed:
            logger.info("Ground station connection closed")
            self._connected = False
        except Exception as e:
            logger.error("Error in message handler: %s", e)
            self._connected = False
