# Performance Guide

## Edge Device Optimization

### Raspberry Pi Configuration

#### Pi Zero 2W (CPU-optimized)
```yaml
# Larger chunks reduce CPU overhead
chunk_size: 1024
chunk_size_ow: 1280

# Higher thresholds reduce false positives  
vad_threshold: 0.7
vad_trigger: 2
wakework_detection_threshold: 0.65

# Shorter timeouts save memory
max_command_input_seconds: 10
```

#### Raspberry Pi 4 (Latency-optimized)
```yaml  
# Smaller chunks for lower latency
chunk_size: 512
chunk_size_ow: 1280

# More sensitive detection
vad_threshold: 0.6
vad_trigger: 1
wakework_detection_threshold: 0.6

# Can handle longer commands
max_command_input_seconds: 15
```

## Performance Monitoring

### Key Metrics
- **CPU Usage**: 15-25% (Pi Zero 2W), 8-15% (Pi 4)
- **Memory**: ~50-100MB baseline, ~150-200MB peak
- **Latency**: Wake word detection ~100-200ms

### Monitoring Commands
```bash
# CPU and memory usage
top -p $(pgrep -f comms-satellite)

# Audio processing health
dmesg | grep -i audio

# Temperature monitoring (Pi)
vcgencmd measure_temp
```

## Optimization Strategies

### Audio Buffer Tuning
- **Lower Latency**: Smaller `chunk_size` (256-512)
- **Lower CPU**: Larger `chunk_size` (1024-2048)
- **Balance**: 512 samples for most use cases

### VAD Sensitivity
- **Faster Response**: Lower `vad_threshold`, `vad_trigger: 1`
- **Fewer False Positives**: Higher `vad_threshold`, `vad_trigger: 2-3`

### Network Optimization
- Use local STT/TTS APIs when possible
- Consider API caching for common responses
- Optimize MQTT broker placement (local network)