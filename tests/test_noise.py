import numpy as np
import pytest

from ligo_pipeline.noise import average_psd, validate_noise_segments


def test_validate_noise_segments_rejects_empty():
    with pytest.raises(ValueError):
        validate_noise_segments({}, expected_duration_sec=10, expected_sample_rate_hz=100)


def test_validate_noise_segments_rejects_wrong_length():
    segments = {0.0: (np.zeros(500), np.arange(500))}  # too short for 10s @ 100Hz = 1000 samples
    with pytest.raises(ValueError):
        validate_noise_segments(segments, expected_duration_sec=10, expected_sample_rate_hz=100)


def test_validate_noise_segments_rejects_times_length_mismatch():
    segments = {0.0: (np.zeros(1000), np.arange(999))}
    with pytest.raises(ValueError):
        validate_noise_segments(segments, expected_duration_sec=10, expected_sample_rate_hz=100)


def test_validate_noise_segments_accepts_well_formed_input():
    segments = {0.0: (np.zeros(1000), np.arange(1000)), 100.0: (np.ones(1000), np.arange(1000))}
    validate_noise_segments(segments, expected_duration_sec=10, expected_sample_rate_hz=100)  # no raise


def test_average_psd_matches_manual_mean_of_constituent_psds():
    sample_rate = 256
    rng = np.random.default_rng(0)
    duration_sec = 8
    n = sample_rate * duration_sec

    segments = {
        0.0: (rng.normal(scale=1e-18, size=n), np.arange(n)),
        1000.0: (rng.normal(scale=1e-18, size=n), np.arange(n)),
    }

    freqs, avg_psd = average_psd(segments, sample_rate)

    from ligo_pipeline.preprocessing import estimate_psd
    individual_psds = [estimate_psd(seg, sample_rate)[1] for seg, _ in segments.values()]
    expected = np.mean(individual_psds, axis=0)

    np.testing.assert_allclose(avg_psd, expected)
    assert freqs.shape == avg_psd.shape
