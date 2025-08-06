# Getting Started Guide

This guide walks you through setting up the Private Assistant Communications Satellite on Ubuntu Server. Follow these steps to get your satellite up and running quickly.

## Prerequisites

- Ubuntu Server 20.04 or newer
- Microphone and speakers (or audio interface)
- Internet connection
- MQTT broker and STT/TTS API endpoints

## Step 1: Install Python 3.12

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 and dependencies
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-dev python3.12-venv git curl

# Install build tools and audio system dependencies
sudo apt install -y gcc build-essential libasound2-dev libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile1-dev

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

## Step 2: Set Up Audio

```bash
# Install audio utilities
sudo apt install -y alsa-utils

# List audio devices
arecord -l   # Input devices
aplay -l     # Output devices

# Set volume levels
alsamixer    # Use arrow keys to adjust, press Esc to exit

# Save volume settings to persist after reboot
sudo alsactl store

# Find sounddevice device indices
python3.12 -c "
import sounddevice as sd
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f'{i}: {device[\"name\"]} (in:{device[\"max_input_channels\"]}, out:{device[\"max_output_channels\"]})')
"
```

Note the device indices for your microphone and speaker.

## Step 3: Create Project and Install Satellite

```bash
# Create project directory
mkdir comms-satellite
cd comms-satellite

# Initialize UV project and install satellite
uv init --python 3.12
uv add private-assistant-comms-satellite

# Test installation
uv run comms-satellite --help
```

## Step 4: Configure

```bash
# Generate configuration template
uv run comms-satellite config-template

# Edit configuration
nano local_config.yaml
```

**Key settings to update:**

```yaml
# Device identification
client_id: "my_satellite"
room: "office"

# Audio devices (from Step 2)
input_device_index: 1    # Your microphone
output_device_index: 1   # Your speaker

# MQTT broker
mqtt_server_host: "your-mqtt-broker.local"
mqtt_server_port: 1883

# APIs 
speech_transcription_api: "http://your-stt-server:8000/transcribe"
speech_synthesis_api: "http://your-tts-server:8080/synthesizeSpeech"
```

## Step 5: Test Run

```bash
# Test manually
uv run comms-satellite start local_config.yaml
```

Say "Hey Nova" followed by a command. Press `Ctrl+C` to stop.

## Step 6: Set Up Service

```bash
# Create service file (adjust paths as needed)
sudo tee /etc/systemd/system/comms-satellite.service > /dev/null <<EOF
[Unit]
Description=Private Assistant Communications Satellite
After=network.target sound.target

[Service]
Type=simple
User=$USER
Group=audio
WorkingDirectory=$HOME/comms-satellite
Environment=PRIVATE_ASSISTANT_API_CONFIG_PATH=$HOME/comms-satellite/local_config.yaml
ExecStart=$HOME/.local/bin/uv run comms-satellite start
Restart=always
RestartSec=10

# Real-time scheduling for audio processing
Nice=-10
IOSchedulingClass=1
IOSchedulingPriority=4

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable comms-satellite.service
sudo systemctl start comms-satellite.service

# Check status
sudo systemctl status comms-satellite.service
```

## Monitor and Troubleshoot

```bash
# View logs
sudo journalctl -fu comms-satellite.service

# Check if running
sudo systemctl is-active comms-satellite.service

# Restart if needed
sudo systemctl restart comms-satellite.service
```

## Common Issues

**Audio errors**: Verify device indices with the sounddevice command from Step 2.

**MQTT errors**: Test connectivity with `ping your-mqtt-broker.local`.

**API errors**: Test endpoints with curl:
```bash
curl -v http://your-stt-server:8000/health
curl -v http://your-tts-server:8080/health
```

That's it! Your satellite should now be running and ready to process voice commands.