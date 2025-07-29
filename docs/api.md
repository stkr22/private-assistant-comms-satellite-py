# API Reference

## Core Classes

### Satellite Class
**Location**: `satellite.py:30`

Main satellite class managing audio processing and MQTT communication.

```python
def start() -> None
    """Start satellite with MQTT client and audio processing."""

def cleanup() -> None  
    """Clean up resources including MQTT client and audio streams."""

def process_output_queue()
    # Processes TTS responses from assistant

def processing_spoken_commands() -> None
    # Handles voice recording after wake word detection
```

### Config Class
**Location**: `utils/config.py:11`

Pydantic configuration model with validation.

```python
# Key configuration parameters
wakework_detection_threshold: float = 0.6
vad_threshold: float = 0.6  # 0-1, 1 is speech
vad_trigger: int = 1       # chunks to cross threshold
chunk_size: int = 512      # audio processing chunk size
```

**Properties**:
- `base_topic` → `assistant/comms_bridge/all/{client_id}`
- `input_topic` → `{base_topic}/input`  
- `output_topic` → `{base_topic}/output`

### SileroVad Class
**Location**: `silero_vad.py:6`

VAD with trigger-level mechanism to reduce false positives.

```python
def __call__(self, audio_bytes: bytes | None) -> bool
    # Returns True only after sustained speech detection
```

## Utility Functions

### Speech Processing
**Location**: `utils/speech_recognition_tools.py`

```python
async def send_audio_to_stt_api(...) -> STTResponse | None
    # Sends audio to transcription API

async def send_text_to_tts_api(...) -> bytes | None  
    # Sends text to synthesis API

def int2float(sound: NDArray[np.int16]) -> NDArray[np.float32]
    # Audio format conversion for API calls
```

### MQTT Client
**Location**: `utils/mqtt_utils.py:11`

```python
async def start() -> None
    # Connects and subscribes to topics

async def publish(topic: str, payload: str, qos: int = 1) -> None
    # Publishes messages to broker
```

## CLI Interface

```bash
comms-satellite [config_path]     # Start satellite
comms-satellite version           # Show version  
comms-satellite config-template   # Generate config template
```