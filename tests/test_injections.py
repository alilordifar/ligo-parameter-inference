import numpy as np
import pytest

from ligo_pipeline.injections import _compute_injection_window, draw_bbh_params
from ligo_pipeline.preprocessing import compute_derived_params


# --- draw_bbh_params --------------------------------------------------

def test_draw_bbh_params_enforces_mass_ordering():
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = draw_bbh_params(rng)
        assert params["mass_1"] >= params["mass_2"]


def test_draw_bbh_params_respects_configured_priors():
    import config
    rng = np.random.default_rng(1)
    m_lo, m_hi = config.CONFIG["mass_prior_msun"]
    d_lo, d_hi = config.CONFIG["distance_prior_mpc"]
    for _ in range(200):
        params = draw_bbh_params(rng)
        assert m_lo <= params["mass_2"] <= params["mass_1"] <= m_hi
        assert d_lo <= params["distance"] <= d_hi


# --- compute_derived_params --------------------------------------------------

def test_compute_derived_params_equal_masses_equal_spins():
    # m1 == m2 == 30, chi1 == chi2 == 0.5 -> chirp_mass == 30 * 2^(-1/5),
    # mass_ratio == 1, chi_eff == 0.5
    params = dict(mass_1=30.0, mass_2=30.0, chi_1=0.5, chi_2=0.5, distance=500.0)
    out = compute_derived_params(params)
    chirp_mass, mass_ratio, chi_eff, distance = out
    assert chirp_mass == pytest.approx(30.0 * 2 ** (-1 / 5))
    assert mass_ratio == pytest.approx(1.0)
    assert chi_eff == pytest.approx(0.5)
    assert distance == pytest.approx(500.0)


def test_compute_derived_params_known_values():
    # hand-computed reference: m1=36, m2=29, chi1=0.3, chi2=-0.1
    params = dict(mass_1=36.0, mass_2=29.0, chi_1=0.3, chi_2=-0.1, distance=410.0)
    chirp_mass, mass_ratio, chi_eff, distance = compute_derived_params(params)
    expected_chirp_mass = (36.0 * 29.0) ** (3 / 5) / (36.0 + 29.0) ** (1 / 5)
    expected_chi_eff = (36.0 * 0.3 + 29.0 * -0.1) / (36.0 + 29.0)
    assert chirp_mass == pytest.approx(expected_chirp_mass)
    assert mass_ratio == pytest.approx(29.0 / 36.0)
    assert chi_eff == pytest.approx(expected_chi_eff)
    assert distance == pytest.approx(410.0)


# --- _compute_injection_window (peak-alignment + buffer sizing) --------------

def test_injection_window_centers_on_merger_via_peak_sample():
    # waveform peak is its 100th sample (of 400); merger should land at
    # merger_idx regardless of where the peak sits inside the waveform.
    sample_rate = 100.0
    window = _compute_injection_window(
        peak_idx=100, waveform_len=400, merger_offset_sec=30.0, sample_rate=sample_rate,
        buffer_sec=5.0, safety_margin_sec=1.0, noise_len=100_000,
    )
    assert window["merger_idx"] == 3000
    # start_idx = merger_idx - peak_idx: the waveform's own sample 0 lands here
    assert window["start_idx"] == 3000 - 100
    # the waveform's peak sample lands exactly at the local merger index
    local_peak_idx = window["local_start"] + 100
    assert local_peak_idx == window["local_merger_idx"]


def test_injection_window_buffer_covers_asymmetric_waveform():
    # peak near the END of a long waveform: the buffer must be wide enough
    # on the LEFT of merger to hold the long pre-merger inspiral.
    sample_rate = 100.0
    window = _compute_injection_window(
        peak_idx=1900, waveform_len=2000, merger_offset_sec=50.0, sample_rate=sample_rate,
        buffer_sec=1.0, safety_margin_sec=0.0, noise_len=1_000_000,
    )
    # buffer half-width must be at least peak_idx (1900 samples pre-merger)
    buf_half_width = window["merger_idx"] - window["buf_start"]
    assert buf_half_width >= 1900
    # the whole waveform must fit inside [buf_start, buf_end)
    assert window["buf_start"] <= window["start_idx"]
    assert window["start_idx"] + 2000 <= window["buf_end"]


def test_injection_window_clips_to_noise_segment_bounds():
    # merger very close to the start of a short noise segment: buf_start
    # must clip to 0, not go negative.
    sample_rate = 100.0
    window = _compute_injection_window(
        peak_idx=50, waveform_len=100, merger_offset_sec=1.0, sample_rate=sample_rate,
        buffer_sec=5.0, safety_margin_sec=0.0, noise_len=10_000,
    )
    assert window["buf_start"] == 0

    # merger very close to the end: buf_end must clip to noise_len, not overshoot.
    window_end = _compute_injection_window(
        peak_idx=50, waveform_len=100, merger_offset_sec=99.9, sample_rate=sample_rate,
        buffer_sec=5.0, safety_margin_sec=0.0, noise_len=10_000,
    )
    assert window_end["buf_end"] == 10_000


def test_injection_window_buffer_at_least_configured_minimum():
    # a tiny waveform should still get at least `buffer_sec` of padding
    # either side of merger, even though half_width alone wouldn't require it.
    sample_rate = 100.0
    window = _compute_injection_window(
        peak_idx=5, waveform_len=10, merger_offset_sec=50.0, sample_rate=sample_rate,
        buffer_sec=5.0, safety_margin_sec=0.0, noise_len=1_000_000,
    )
    buf_half_width = window["merger_idx"] - window["buf_start"]
    assert buf_half_width >= 5.0 * sample_rate
