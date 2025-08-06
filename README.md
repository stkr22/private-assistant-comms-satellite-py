# Private Assistant Communications Satellite

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**A high-performance voice interaction satellite optimized for edge devices like Raspberry Pi**

The Private Assistant Communications Satellite is a latency-optimized edge device component that provides voice interaction capabilities for the private assistant ecosystem. Designed specifically for resource-constrained devices like Raspberry Pi Zero 2W and Raspberry Pi 4, it delivers real-time wake word detection, voice activity detection, and seamless integration with speech processing APIs.

## 🎯 Key Features

- **🔊 Real-time Wake Word Detection** - OpenWakeWord integration with customizable models
- **🎤 Voice Activity Detection (VAD)** - Silero VAD for accurate speech boundary detection  
- **🗣️ Speech Processing Integration** - Async STT/TTS API communication
- **📡 MQTT Ecosystem Integration** - Low-latency message passing with the assistant ecosystem
- **⚡ Edge Device Optimized** - Multi-threaded architecture minimizing audio processing latency
- **🔧 Flexible Configuration** - YAML-based configuration with environment variable support
- **🛡️ Robust Error Handling** - Graceful degradation and automatic recovery

## 🏗️ Architecture Overview

The satellite uses a **simple state machine architecture** for stability and performance:

```
┌─────────────────┐    ┌─────────────────┐
│   Main Thread   │    │   MQTT Thread   │
│ (State Machine) │    │  (Low Latency)  │
├─────────────────┤    ├─────────────────┤
│ • LISTENING     │    │ • MQTT Client   │
│ • RECORDING     │◄──►│ • Message Queue │
│ • WAITING       │    │ • Event Loop    │
│ • SPEAKING      │    │ • Async I/O     │
└─────────────────┘    └─────────────────┘
```

**State Machine Flow:**
- **LISTENING**: Monitors audio for wake word detection
- **RECORDING**: Records user speech after wake word trigger
- **WAITING**: Processes STT API and waits for response
- **SPEAKING**: Plays TTS audio response

**Key Design Benefits:**
- **Simplified Threading**: Only MQTT runs in separate thread for network I/O
- **Predictable Behavior**: Clear state transitions eliminate race conditions  
- **Resource Efficiency**: Callback-based audio I/O with efficient buffering
- **Stable Operation**: No queue overflows or threading conflicts

## 🚀 Quick Start

### System Requirements

**Minimum Hardware:**
- Raspberry Pi Zero 2W (1GB RAM) or better
- USB microphone or I2S audio HAT
- Speaker or headphones for audio feedback
- MicroSD card (16GB+ recommended)

**Recommended Hardware:**
- Raspberry Pi 4 (2GB+ RAM) for optimal performance  
- Quality USB microphone with noise cancellation
- Dedicated audio HAT for better audio quality

### Installation

#### 1. System Dependencies (Raspberry Pi OS)

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install audio system dependencies
sudo apt-get install -y \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1-dev \
    python3.12-dev \
    git

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

#### 2. Install Satellite

```bash
# Clone repository
git clone https://github.com/stkr22/private-assistant-comms-satellite-py.git
cd private-assistant-comms-satellite-py

# Install dependencies (development)
uv sync --group dev

# Install for deployment (all dependencies included)
uv sync
```

#### 3. Quick Configuration

```bash
# Generate configuration template
uv run comms-satellite config-template

# Edit configuration for your setup
nano local_config.yaml
```

#### 4. Test Installation

```bash
# Test without audio (verify dependencies)
uv run python -c "import private_assistant_comms_satellite; print('✅ Installation successful')"

# Test with audio (requires microphone)
uv run comms-satellite --help
```

## ⚙️ Configuration

### Essential Configuration

The satellite requires several key configuration parameters. Generate a template with:

```bash
uv run comms-satellite config-template
```

**Minimum Required Configuration:**

```yaml
# Wake word settings
wakework_detection_threshold: 0.6
path_or_name_wakeword_model: "./assets/wakeword_models/hey_nova.onnx"
name_wakeword_model: "hey_nova"

# API endpoints - REQUIRED
speech_transcription_api: "http://your-stt-server:8000/transcribe"
speech_synthesis_api: "http://your-tts-server:8080/synthesizeSpeech"

# Device identification
client_id: "living_room_satellite"  # Unique name for this device
room: "livingroom"

# MQTT broker - REQUIRED  
mqtt_server_host: "your-mqtt-broker.local"
mqtt_server_port: 1883

# Audio device indices (use 'python -c "import sounddevice; print(sounddevice.query_devices())"' to list devices)
input_device_index: 1   # Your microphone
output_device_index: 1  # Your speaker
```

### Performance Tuning by Device

#### Raspberry Pi Zero 2W Configuration
```yaml
# Optimize for minimal CPU usage
chunk_size: 1024        # Larger chunks = less CPU overhead
chunk_size_ow: 1280     # OpenWakeWord optimized size
samplerate: 16000       # Standard rate, good quality/performance balance
vad_threshold: 0.7      # Higher threshold = less false positives
vad_trigger: 2          # Require 2 chunks before activation
max_command_input_seconds: 10  # Shorter timeout saves memory
```

#### Raspberry Pi 4 Configuration  
```yaml
# Optimize for lower latency
chunk_size: 512         # Smaller chunks = lower latency
chunk_size_ow: 1280     # Keep OpenWakeWord optimized
samplerate: 16000       # Can handle higher if needed
vad_threshold: 0.6      # More sensitive detection
vad_trigger: 1          # Immediate activation
max_command_input_seconds: 15  # Longer timeout for complex commands
```

### Audio Device Configuration

```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone input
arecord -l  # List recording devices
arecord -D hw:1,0 -f cd test.wav  # Test recording

# Test speaker output  
aplay -l   # List playback devices
aplay -D hw:1,0 /usr/share/sounds/alsa/Front_Left.wav  # Test playback
```

## 🎮 Usage

### Basic Usage

```bash
# Start with default configuration
uv run comms-satellite local_config.yaml

# Start with custom configuration
uv run comms-satellite /path/to/my_config.yaml

# Use environment variable
export PRIVATE_ASSISTANT_API_CONFIG_PATH="/path/to/config.yaml"
uv run comms-satellite
```

### Command Line Options

```bash
# Show version
uv run comms-satellite version

# Generate configuration template
uv run comms-satellite config-template

# Help
uv run comms-satellite --help
```

### Systemd Service (Recommended for Production)

Create a systemd service for automatic startup:

```bash
# Create service file
sudo tee /etc/systemd/system/satellite.service > /dev/null <<EOF
[Unit]
Description=Private Assistant Communications Satellite
After=network.target sound.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=audio
WorkingDirectory=/home/pi/private-assistant-comms-satellite-py
Environment=PRIVATE_ASSISTANT_API_CONFIG_PATH=/home/pi/satellite-config.yaml
ExecStart=/home/pi/.local/bin/uv run comms-satellite
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable satellite.service
sudo systemctl start satellite.service

# Check status
sudo systemctl status satellite.service
sudo journalctl -fu satellite.service
```

## 🔌 API Integration

### Speech-to-Text (STT) API

The satellite expects an STT service that accepts audio data and returns transcribed text.

**Expected API Contract:**
```http
POST /transcribe
Content-Type: multipart/form-data
Authorization: Bearer <optional-token>

Body: 
- file: audio.raw (raw PCM audio data, 16kHz, 16-bit, mono)

Response:
{
  "text": "transcribed speech text",
  "message": "success"
}
```

**Configuration:**
```yaml
speech_transcription_api: "http://your-stt-server:8000/transcribe"
speech_transcription_api_token: "optional-bearer-token"
```

### Text-to-Speech (TTS) API

The satellite expects a TTS service that accepts text and returns audio data.

**Expected API Contract:**
```http
POST /synthesizeSpeech
Content-Type: application/json
Authorization: Bearer <optional-token>

Body:
{
  "text": "text to synthesize",
  "sample_rate": 16000
}

Response: audio/wav (PCM audio data)
```

**Configuration:**
```yaml
speech_synthesis_api: "http://your-tts-server:8080/synthesizeSpeech"  
speech_synthesis_api_token: "optional-bearer-token"
```

## 📡 MQTT Integration

### Topic Structure

The satellite uses a hierarchical MQTT topic structure:

```
assistant/comms_bridge/all/{client_id}/
├── input          # Satellite publishes recognized speech here
└── output         # Satellite subscribes for responses to speak

assistant/comms_bridge/broadcast  # System-wide announcements
```

**Topic Configuration:**
```yaml
# Default topics (auto-generated)
client_id: "living_room_satellite"
# Results in: assistant/comms_bridge/all/living_room_satellite/

# Custom topic overrides
base_topic_overwrite: "custom/satellite/living_room"
input_topic_overwrite: "custom/satellite/living_room/speech_input"
output_topic_overwrite: "custom/satellite/living_room/speech_output"
```

### Message Formats

**Input Messages (Satellite → Assistant):**
```json
{
  "id": "uuid4-string",
  "text": "recognized speech text",
  "room": "livingroom", 
  "output_topic": "assistant/comms_bridge/all/living_room_satellite/output"
}
```

**Output Messages (Assistant → Satellite):**
```json
{
  "text": "response text to speak",
  "id": "uuid4-string"
}
```

## 🎛️ Performance Optimization

### Latency Optimization

**Key Performance Metrics:**
- Wake word detection: ~100-200ms
- Voice activity detection: ~50-100ms per chunk
- STT API call: 1-5 seconds (network dependent)
- TTS API call: 0.5-2 seconds (network dependent)

**Optimization Strategies:**

1. **Audio Buffer Tuning:**
```yaml
# Lower latency (higher CPU usage)
chunk_size: 256
chunk_size_ow: 1280

# Higher latency (lower CPU usage)  
chunk_size: 1024
chunk_size_ow: 1280
```

2. **VAD Sensitivity Tuning:**
```yaml
# More sensitive (faster activation, more false positives)
vad_threshold: 0.5
vad_trigger: 1

# Less sensitive (slower activation, fewer false positives)
vad_threshold: 0.8
vad_trigger: 3
```

3. **Network Optimization:**
```yaml
# Local APIs for best performance
speech_transcription_api: "http://localhost:8000/transcribe"
speech_synthesis_api: "http://localhost:8080/synthesizeSpeech"
```

### Memory Usage Optimization

**For Memory-Constrained Devices (Pi Zero 2W):**
```yaml
max_command_input_seconds: 8    # Limit recording buffer
samplerate: 16000              # Don't increase sample rate unnecessarily
chunk_size: 1024               # Larger chunks = fewer allocations
```

**Monitor Memory Usage:**
```bash
# Monitor system memory
watch -n 1 'free -h && ps aux | grep satellite | head -5'

# Monitor satellite process specifically  
top -p $(pgrep -f comms-satellite)
```

## 🔧 Troubleshooting

### Common Issues

#### Audio Device Problems
```bash
# Check audio devices
python -c "import sounddevice as sd; print(f'Devices: {len(sd.query_devices())}')"

# Test microphone
arecord -D hw:1,0 -f cd -t wav -r 16000 test.wav

# Check ALSA configuration
cat /proc/asound/cards
```

#### MQTT Connection Issues
```bash
# Test MQTT connectivity
mosquitto_pub -h your-mqtt-broker -t "test/topic" -m "test message"
mosquitto_sub -h your-mqtt-broker -t "assistant/comms_bridge/all/+/+"

# Check network connectivity
ping your-mqtt-broker
telnet your-mqtt-broker 1883
```

#### API Integration Issues
```bash
# Test STT API manually
curl -X POST -F "file=@test.wav" \
  -H "user-token: your-token" \
  http://your-stt-server:8000/transcribe

# Test TTS API manually
curl -X POST -H "Content-Type: application/json" \
  -H "user-token: your-token" \
  -d '{"text":"hello world","sample_rate":16000}' \
  http://your-tts-server:8080/synthesizeSpeech > test_output.wav
```

#### Performance Issues
```bash
# Check CPU usage
htop

# Check system load
uptime

# Monitor audio dropouts
dmesg | grep -i audio

# Check for thermal throttling (Raspberry Pi)
vcgencmd measure_temp
vcgencmd get_throttled
```

### Debug Mode

Enable detailed logging by setting environment variable:
```bash
export PYTHONPATH=src
export LOG_LEVEL=DEBUG
uv run comms-satellite config.yaml
```

## 👨‍💻 Development

### Development Setup

```bash
# Clone repository
git clone https://github.com/stkr22/private-assistant-comms-satellite-py.git
cd private-assistant-comms-satellite-py

# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality Tools

```bash
# Type checking
uv run mypy src/

# Linting  
uv run ruff check .

# Code formatting
uv run ruff format .

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=private_assistant_comms_satellite --cov-report=html
```

### Project Structure

```
private-assistant-comms-satellite-py/
├── src/private_assistant_comms_satellite/
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface
│   ├── main.py                  # Application entry point
│   ├── satellite.py            # Core Satellite class (main logic)
│   ├── silero_vad.py           # Voice Activity Detection
│   └── utils/
│       ├── config.py           # Configuration management
│       ├── mqtt_utils.py       # MQTT client wrapper
│       └── speech_recognition_tools.py  # STT/TTS API integration
├── tests/                       # Test suite
├── docs/                        # Documentation
├── assets/                      # Wake word models and audio files
└── pyproject.toml              # Project configuration
```

### Contributing

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Follow code style:** Run `uv run ruff format .` and `uv run ruff check .`
4. **Add tests:** Ensure new features have appropriate test coverage
5. **Commit changes:** Use conventional commit format: `feat: add amazing feature [AI]`
6. **Push to branch:** `git push origin feature/amazing-feature`
7. **Create Pull Request**

**Code Style Guidelines:**
- Follow the existing code style (enforced by Ruff)
- Add type hints to all public functions
- Include docstrings for public classes and methods
- Add AIDEV-* anchor comments for significant code sections
- Keep functions focused and under 50 lines when possible

## 📚 Additional Resources

- **[Architecture Documentation](docs/architecture.md)** - Detailed system design
- **[API Reference](docs/api.md)** - Complete API documentation  
- **[Performance Guide](docs/performance.md)** - In-depth optimization techniques
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Extended troubleshooting
- **[Contributing Guide](CONTRIBUTING.md)** - Detailed contribution guidelines

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenWakeWord** - Wake word detection engine
- **Silero VAD** - Voice activity detection
- **Private Assistant Commons** - Shared utilities and message formats
- **sounddevice** - Python audio I/O library with NumPy integration
- **aiomqtt** - Async MQTT client library
