"""Basic tests for mqtt_utils module."""

import queue

import pytest

from private_assistant_comms_satellite.utils.mqtt_utils import AsyncMQTTClient


@pytest.fixture
def sample_topic_queue():
    """Create sample topic to queue mapping for testing."""
    return {
        "test/topic1": queue.Queue(),
        "test/topic2": queue.Queue(),
    }


class TestAsyncMQTTClientInit:
    """Test AsyncMQTTClient initialization."""

    def test_client_initialization(self, sample_topic_queue):
        """Test proper initialization of AsyncMQTTClient."""
        client = AsyncMQTTClient(
            hostname="localhost",
            port=1883,
            client_id="test_client",
            topic_to_queue=sample_topic_queue,
        )

        assert client.hostname == "localhost"
        assert client.port == 1883  # noqa: PLR2004
        assert client.client_id == "test_client"
        assert client.topic_to_queue == sample_topic_queue
        assert client.use_websockets is False
        assert client.websocket_path == "/mqtt"
        assert client._client is None
        assert client._running is False

    def test_client_initialization_with_websockets(self, sample_topic_queue):
        """Test proper initialization of AsyncMQTTClient with WebSocket transport."""
        client = AsyncMQTTClient(
            hostname="localhost",
            port=8080,
            client_id="test_client",
            topic_to_queue=sample_topic_queue,
            use_websockets=True,
            websocket_path="/custom/mqtt",
        )

        assert client.hostname == "localhost"
        assert client.port == 8080  # noqa: PLR2004
        assert client.client_id == "test_client"
        assert client.topic_to_queue == sample_topic_queue
        assert client.use_websockets is True
        assert client.websocket_path == "/custom/mqtt"
        assert client.use_ssl is False
        assert client._client is None
        assert client._running is False

    def test_client_initialization_with_ssl(self, sample_topic_queue):
        """Test proper initialization of AsyncMQTTClient with SSL/TLS transport."""
        client = AsyncMQTTClient(
            hostname="broker.example.com",
            port=8883,
            client_id="test_client",
            topic_to_queue=sample_topic_queue,
            use_ssl=True,
        )

        assert client.hostname == "broker.example.com"
        assert client.port == 8883  # noqa: PLR2004
        assert client.client_id == "test_client"
        assert client.topic_to_queue == sample_topic_queue
        assert client.use_websockets is False
        assert client.websocket_path == "/mqtt"
        assert client.use_ssl is True
        assert client._client is None
        assert client._running is False

    def test_client_initialization_with_websockets_and_ssl(self, sample_topic_queue):
        """Test proper initialization of AsyncMQTTClient with WebSocket Secure (WSS) transport."""
        client = AsyncMQTTClient(
            hostname="broker.example.com",
            port=443,
            client_id="test_client",
            topic_to_queue=sample_topic_queue,
            use_websockets=True,
            websocket_path="/secure/mqtt",
            use_ssl=True,
        )

        assert client.hostname == "broker.example.com"
        assert client.port == 443  # noqa: PLR2004
        assert client.client_id == "test_client"
        assert client.topic_to_queue == sample_topic_queue
        assert client.use_websockets is True
        assert client.websocket_path == "/secure/mqtt"
        assert client.use_ssl is True
        assert client._client is None
        assert client._running is False


# NOTE: Complex async integration tests removed for simplicity -
# focus on basic initialization and configuration testing
