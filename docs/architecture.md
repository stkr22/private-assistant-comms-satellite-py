# Architecture

## Threading Model

The satellite uses a **2-thread architecture** optimized for low-latency audio processing:

```
┌─────────────────┐    ┌─────────────────┐
│   Main Thread   │    │   MQTT Thread   │
│ (Audio/Real-time)│    │  (Network I/O)  │
├─────────────────┤    ├─────────────────┤
│ • PyAudio I/O   │    │ • MQTT Client   │
│ • Wake Word     │◄──►│ • STT/TTS APIs  │
│ • VAD Processing│    │ • Message Queue │
│ • Sound Playback│    │ • Async Event   │
└─────────────────┘    │   Loop          │
                       └─────────────────┘
```

**Key Design Decision**: Separate threads prevent network I/O from blocking audio processing, critical for edge device performance.

## Data Flow

### Audio Processing Pipeline
```
Microphone → PyAudio → Wake Word Detection → VAD → STT API → MQTT → Assistant
                              ↓
                        Start/Stop Sounds

Assistant → MQTT → TTS API → PyAudio → Speaker
```

### Configuration Structure
- **Pydantic Models**: Type-safe configuration with validation
- **Dynamic Topics**: MQTT topics generated from `client_id`
- **Performance Tuning**: Configurable chunk sizes and thresholds

## Performance Optimization

### Edge Device Tuning
```yaml
# Pi Zero 2W (CPU-optimized)
chunk_size: 1024
vad_threshold: 0.7
vad_trigger: 2

# Pi 4 (Latency-optimized)  
chunk_size: 512
vad_threshold: 0.6
vad_trigger: 1
```

### Memory Management
- Pre-loaded audio files (start/stop sounds)
- Reusable NumPy buffers
- Single ML model instances