# Architecture

## State Machine Model

The satellite uses a **simple state machine** with minimal threading for stability:

```
┌─────────────────┐    ┌─────────────────┐
│   Main Thread   │    │   MQTT Thread   │
│ (State Machine) │    │  (Network I/O)  │
├─────────────────┤    ├─────────────────┤
│ • LISTENING     │    │ • MQTT Client   │
│ • RECORDING     │◄──►│ • Message Queue │
│ • WAITING       │    │ • Event Loop    │
│ • SPEAKING      │    │ • Low Latency   │
└─────────────────┘    └─────────────────┘
```

**Key Design Benefits**: Single-threaded state machine eliminates race conditions and queue overflows while maintaining MQTT responsiveness.

## State Transitions

### State Flow
```
LISTENING ──(wake word)──► RECORDING ──(silence)──► WAITING ──(STT done)──► LISTENING
    ▲                                                    │
    │                                               (TTS received)
    │                                                    ▼
    └──────────────────────────────────────────── SPEAKING
```

### State Behaviors
- **LISTENING**: Wake word detection, MQTT processing
- **RECORDING**: Voice recording with VAD
- **WAITING**: STT processing, MQTT processing  
- **SPEAKING**: TTS playback, MQTT processing

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