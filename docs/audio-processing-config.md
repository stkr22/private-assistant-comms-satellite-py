# Audio Processing Configuration

The satellite uses a two-stage audio enhancement pipeline optimized for wake word detection:
1. **Parametric EQ** - Boosts speech clarity (3.5 kHz presence)
2. **Automatic Gain Control (AGC)** - Normalizes volume levels

**Important**: Filters are applied BEFORE wake word detection. Modifications affect MicroWakeWord's ability to detect patterns.

## Audio Pipeline Flow

```
sounddevice → Parametric EQ → AGC → int16 → MicroFrontend (spectrogram) → TFLite inference → VAD → Recording
```

## Parametric EQ for Voice Enhancement

Boosts critical speech frequencies to improve intelligibility and reduce hollow sound.

### Configuration Options

```yaml
audio_processing:
  enable_voice_eq: true              # Enable/disable EQ
  eq_presence_boost_db: 3.0          # Presence boost (0-6 dB) - OPTIMIZED for detection
  eq_presence_freq_hz: 3500.0        # Center frequency (2000-5000 Hz)
  eq_presence_q: 2.5                 # Q factor (0.5-5.0, higher = narrower)
```

### Tuning Guide

**For thin/hollow voice**: Increase presence boost to 4-5 dB
**For nasal voice**: Lower presence frequency to 2500-3000 Hz
**For minimal processing**: Set presence boost to 1-2 dB

**Note**: This EQ only boosts mid-high frequencies (presence) which are critical for wake word detection. MicroWakeWord's MicroFrontend has built-in PCAN noise suppression at the feature level.

### Performance Impact

- **CPU**: +1-3% on Raspberry Pi Zero 2 W
- **Memory**: +1 MB (negligible)
- **Latency**: <1ms per chunk (imperceptible)

## Automatic Gain Control (AGC)

Normalizes volume levels and boosts quiet speech.

### Configuration Options

```yaml
audio_processing:
  enable_agc: true                   # ENABLED by default (optimized)
  agc_target_rms: 0.16               # Target RMS level (0.01-0.5) - OPTIMIZED
  agc_smoothing: 0.7                 # Smoothing factor (0-0.99) - OPTIMIZED
```

### Recommended Settings

**Default (Optimized)**: `target_rms=0.16`, `smoothing=0.7` ✅
**Louder output**: `target_rms=0.2-0.25`
**Quieter/safer**: `target_rms=0.12-0.15`
**Smoother transitions**: `smoothing=0.85-0.9`
**Faster response**: `smoothing=0.5-0.6`

### Performance Impact

- **CPU**: +1% on Raspberry Pi Zero 2 W
- **Memory**: Negligible
- **Latency**: <1ms per chunk (imperceptible)

### Troubleshooting

**Audio sounds "pumping" (volume fluctuates)**:
- Increase smoothing factor (try 0.95)
- Reduce target RMS (try 0.08)

**AGC not boosting enough**:
- Increase target RMS (try 0.15-0.2)
- Check if voice is already near target level

## Testing Audio Quality

Enable debug mode to save audio samples:

```bash
export DEBUG_AUDIO=1
uv run python -m private_assistant_comms_satellite
```

Audio files saved to `/tmp/audio_debug/`:
- `filter_original_*.wav` - Before processing
- `filter_processed_*.wav` - After EQ and AGC

Compare using audio analysis tools or by listening.

## Combined Pipeline Performance

When using both processors together:
- **Total CPU**: +2-4% (EQ + AGC)
- **Total Memory**: +1-2 MB
- **Total Latency**: <2ms (imperceptible)

Well within Raspberry Pi Zero 2 W capabilities (typical baseline 10-20% CPU).
