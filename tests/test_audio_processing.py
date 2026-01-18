"""Tests for audio signal processing."""

import numpy as np

from private_assistant_comms_satellite.audio_processing import ParametricEQ, SimpleAGC


class TestParametricEQ:
    """Test parametric EQ functionality."""

    def test_eq_initialization(self):
        """Test EQ can be initialized with valid parameters."""
        eq = ParametricEQ(sample_rate=16000, presence_boost_db=2.5, presence_freq_hz=3500.0)
        assert eq.sample_rate == 16000  # noqa: PLR2004
        assert len(eq.filters) > 0  # At least presence filter

    def test_eq_boosts_presence_frequency(self):
        """Test EQ boosts presence frequencies."""
        eq = ParametricEQ(sample_rate=16000, presence_boost_db=3.0, presence_freq_hz=3500.0)

        # Create 3.5kHz tone
        duration = 1.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, endpoint=False)
        signal_3500hz = np.sin(2 * np.pi * 3500 * t).astype(np.float32)

        # Process
        boosted = eq.process(signal_3500hz)

        # Should have increased RMS after settling
        original_rms = np.sqrt(np.mean(signal_3500hz[1000:] ** 2))
        boosted_rms = np.sqrt(np.mean(boosted[1000:] ** 2))

        # Should show boost
        assert boosted_rms > original_rms

    def test_eq_streaming_continuity(self):
        """Test EQ maintains state across chunks."""
        eq = ParametricEQ(sample_rate=16000, presence_boost_db=2.5, presence_freq_hz=3500.0)

        # Use deterministic signal (sine wave)
        chunk_size = 512
        num_chunks = 10
        sample_rate = 16000
        frequency = 1000  # Hz

        chunks_processed = []
        for i in range(num_chunks):
            # Generate continuous sine wave chunks
            t_start = i * chunk_size / sample_rate
            t_end = (i + 1) * chunk_size / sample_rate
            t = np.linspace(t_start, t_end, chunk_size, endpoint=False)
            chunk = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5
            filtered_chunk = eq.process(chunk)
            chunks_processed.append(filtered_chunk)

        # Concatenate chunks
        full_filtered = np.concatenate(chunks_processed)

        # Check for discontinuities at chunk boundaries
        boundary_derivatives = []
        internal_derivatives = []

        for i in range(1, num_chunks):
            boundary_idx = i * chunk_size
            # Derivative at boundary
            boundary_deriv = abs(full_filtered[boundary_idx] - full_filtered[boundary_idx - 1])
            boundary_derivatives.append(boundary_deriv)

            # Average derivative around boundary
            for j in range(-10, 10):
                if 0 < boundary_idx + j < len(full_filtered) - 1:
                    deriv = abs(full_filtered[boundary_idx + j + 1] - full_filtered[boundary_idx + j])
                    internal_derivatives.append(deriv)

        # Boundary derivatives should not be significantly larger
        avg_boundary = np.mean(boundary_derivatives)
        avg_internal = np.mean(internal_derivatives)

        assert avg_boundary < avg_internal * 3  # Lenient tolerance

    def test_eq_reset(self):
        """Test EQ state reset functionality."""
        eq = ParametricEQ(sample_rate=16000, presence_boost_db=2.5, presence_freq_hz=3500.0)

        # Process some audio
        audio = np.random.randn(1000).astype(np.float32)
        eq.process(audio)

        # State should be initialized
        assert any(state is not None for state in eq.filter_states)

        # Reset
        eq.reset()

        # State should be cleared
        assert all(state is None for state in eq.filter_states)


class TestSimpleAGC:
    """Test AGC functionality."""

    def test_agc_initialization(self):
        """Test AGC can be initialized."""
        agc = SimpleAGC(target_rms=0.1, smoothing=0.9)
        assert agc.target_rms == 0.1  # noqa: PLR2004
        assert agc.current_gain == 1.0

    def test_agc_normalizes_quiet_signal(self):
        """Test AGC boosts quiet signals."""
        agc = SimpleAGC(target_rms=0.1, smoothing=0.5)  # Lower smoothing for faster response

        # Quiet signal (RMS ~0.01)
        quiet_signal = np.random.randn(16000).astype(np.float32) * 0.01

        # Process in chunks
        chunk_size = 512
        for i in range(0, len(quiet_signal), chunk_size):
            chunk = quiet_signal[i : i + chunk_size]
            if len(chunk) == chunk_size:
                _ = agc.process(chunk)

        # After settling, gain should be > 1 (boosting)
        assert agc.current_gain > 1.0

    def test_agc_attenuates_loud_signal(self):
        """Test AGC reduces loud signals."""
        agc = SimpleAGC(target_rms=0.1, smoothing=0.5)

        # Loud signal (RMS ~0.5)
        loud_signal = np.random.randn(16000).astype(np.float32) * 0.5

        # Process in chunks
        chunk_size = 512
        for i in range(0, len(loud_signal), chunk_size):
            chunk = loud_signal[i : i + chunk_size]
            if len(chunk) == chunk_size:
                _ = agc.process(chunk)

        # After settling, gain should be < 1 (attenuating)
        assert agc.current_gain < 1.0

    def test_agc_reset(self):
        """Test AGC state reset."""
        agc = SimpleAGC(target_rms=0.1, smoothing=0.9)

        # Process some audio
        audio = np.random.randn(1000).astype(np.float32) * 0.01
        for i in range(0, len(audio), 512):
            chunk = audio[i : i + 512]
            if len(chunk) == 512:  # noqa: PLR2004
                agc.process(chunk)

        # Gain should have changed
        assert agc.current_gain != 1.0

        # Reset
        agc.reset()

        # Gain should be reset to 1.0
        assert agc.current_gain == 1.0
