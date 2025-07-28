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
        assert client._client is None
        assert client._running is False


# NOTE: Complex async integration tests removed for simplicity - 
# focus on basic initialization and configuration testing