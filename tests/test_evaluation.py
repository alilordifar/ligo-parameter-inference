import numpy as np
import pytest

from ligo_pipeline.evaluation import detect_psd_drift, summarize_sbc_ks, validate_sbc_ranks


# --- validate_sbc_ranks --------------------------------------------------

def test_validate_sbc_ranks_accepts_correct_shape():
    ranks = np.zeros((10, 4))
    validate_sbc_ranks(ranks, n_examples_expected=10, n_params_expected=4)  # no raise


def test_validate_sbc_ranks_rejects_wrong_shape():
    ranks = np.zeros((10, 3))
    with pytest.raises(ValueError):
        validate_sbc_ranks(ranks, n_examples_expected=10, n_params_expected=4)


def test_validate_sbc_ranks_rejects_nan():
    ranks = np.zeros((5, 2))
    ranks[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_sbc_ranks(ranks, n_examples_expected=5, n_params_expected=2)


# --- summarize_sbc_ks ----------------------------------------------------

def test_summarize_sbc_ks_uniform_ranks_pass_ks_test():
    rng = np.random.default_rng(0)
    num_posterior_samples = 1000
    # perfectly-calibrated model: ranks uniform over [0, num_posterior_samples]
    ranks = rng.integers(0, num_posterior_samples, size=(500, 2)).astype(float)

    summary = summarize_sbc_ks(ranks, num_posterior_samples, ["a", "b"], label="test")
    for name in ("a", "b"):
        # not a hard guarantee for any single seed, but a large uniform sample
        # should not look wildly miscalibrated
        assert summary[name]["p_value"] > 0.01


def test_summarize_sbc_ks_detects_non_uniform_ranks():
    num_posterior_samples = 1000
    # badly miscalibrated: ranks clustered near 0 instead of uniform
    ranks = np.zeros((200, 1)) + 5.0

    summary = summarize_sbc_ks(ranks, num_posterior_samples, ["a"], label="test")
    assert summary["a"]["p_value"] < 0.01


# --- detect_psd_drift ------------------------------------------------------

def _synthetic_asd(freqs, level, rng, noise_frac=0.0):
    base = np.full_like(freqs, level)
    if noise_frac:
        base = base * (1 + rng.normal(scale=noise_frac, size=freqs.shape))
    return base ** 2  # PSD = ASD^2


def test_detect_psd_drift_identical_curves_not_flagged():
    rng = np.random.default_rng(0)
    freqs = np.linspace(10, 500, 200)
    psd = _synthetic_asd(freqs, level=1e-23, rng=rng, noise_frac=0.02)

    result = detect_psd_drift(freqs, psd, freqs, psd, band_low=35, band_high=350)
    assert result["median_relative_asd_diff"] == pytest.approx(0.0, abs=1e-9)
    assert result["flagged_retrain_warranted"] is False


def test_detect_psd_drift_large_shift_flagged_by_magnitude():
    rng = np.random.default_rng(1)
    freqs = np.linspace(10, 500, 200)
    psd_a = _synthetic_asd(freqs, level=1e-23, rng=rng, noise_frac=0.0)
    psd_b = _synthetic_asd(freqs, level=2e-23, rng=rng, noise_frac=0.0)  # ASD doubled

    result = detect_psd_drift(freqs, psd_a, freqs, psd_b, band_low=35, band_high=350,
                               relative_threshold=0.20)
    assert result["median_relative_asd_diff"] == pytest.approx(1.0, rel=1e-6)
    assert result["magnitude_flag"] is True
    assert result["flagged_retrain_warranted"] is True


def test_detect_psd_drift_small_shift_below_threshold_not_flagged():
    rng = np.random.default_rng(2)
    freqs = np.linspace(10, 500, 200)
    psd_a = _synthetic_asd(freqs, level=1e-23, rng=rng, noise_frac=0.0)
    psd_b = _synthetic_asd(freqs, level=1.05e-23, rng=rng, noise_frac=0.0)  # +5% ASD

    result = detect_psd_drift(freqs, psd_a, freqs, psd_b, band_low=35, band_high=350,
                               relative_threshold=0.20, ks_alpha=0.05)
    assert result["magnitude_flag"] is False
