"""Simulated BBH signal injection: parameter draws from the prior, waveform
generation, matched-filter SNR gating, and placement into a real noise
segment around a requested merger time.
"""
import numpy as np

import config
from .preprocessing import estimate_psd
from .utils import logger, validate_finite, validate_strain


class WaveformGenerationError(RuntimeError):
    """Raised when the waveform approximant cannot generate a waveform for a
    given parameter draw. Distinct from an SNR-based rejection (None) so
    callers can count/log the two separately."""


def draw_bbh_params(rng: np.random.Generator) -> dict:
    m_lo, m_hi = config.CONFIG["mass_prior_msun"]
    s_lo, s_hi = config.CONFIG["spin_prior"]
    d_lo, d_hi = config.CONFIG["distance_prior_mpc"]

    mass_1 = rng.uniform(m_lo, m_hi)
    mass_2 = rng.uniform(m_lo, m_hi)
    if mass_2 > mass_1:
        mass_1, mass_2 = mass_2, mass_1  # enforce mass_1 >= mass_2 convention

    chi_1 = rng.uniform(s_lo, s_hi)
    chi_2 = rng.uniform(s_lo, s_hi)

    # comoving-volume-weighted distance draw (uniform in volume, not in distance)
    u = rng.uniform(0, 1)
    distance = (u * (d_hi ** 3 - d_lo ** 3) + d_lo ** 3) ** (1 / 3)

    return dict(mass_1=mass_1, mass_2=mass_2, chi_1=chi_1, chi_2=chi_2, distance=distance)


def _compute_injection_window(peak_idx: int, waveform_len: int, merger_offset_sec: float,
                               sample_rate: float, buffer_sec: float, safety_margin_sec: float,
                               noise_len: int) -> dict:
    """Pure placement/buffer-sizing math for `generate_injection`, factored
    out so it's testable without a waveform generator.

    Merger alignment uses the waveform's PEAK sample, not its last sample --
    aligning on the last sample instead silently biases the merger time by
    however far the ringdown tail extends past the peak. The buffer around
    merger must be wide enough to hold the whole waveform either side of the
    peak, with a small safety margin, and is clipped to the noise segment's
    own bounds.
    """
    merger_idx = int(merger_offset_sec * sample_rate)
    start_idx = merger_idx - peak_idx

    safety_margin = int(safety_margin_sec * sample_rate)
    half_width = max(peak_idx, waveform_len - peak_idx) + safety_margin
    buf_samples = max(int(buffer_sec * sample_rate), half_width)

    buf_start = max(0, merger_idx - buf_samples)
    buf_end = min(noise_len, merger_idx + buf_samples)

    return dict(
        merger_idx=merger_idx,
        start_idx=start_idx,
        buf_start=buf_start,
        buf_end=buf_end,
        local_start=start_idx - buf_start,
        local_merger_idx=merger_idx - buf_start,
    )


def generate_injection(params: dict, noise_seg: np.ndarray, sample_rate: float, merger_offset_sec: float,
                        f_lower: float, snr_threshold: float, buffer_sec: float, safety_margin_sec: float):
    """Generate a waveform, matched-filter SNR it against `noise_seg`'s own
    PSD, and inject it into a buffer around the requested merger time if it
    clears `snr_threshold`. Returns None (an expected rejection, not an
    error) for a below-threshold draw. Raises WaveformGenerationError if the
    approximant can't generate a waveform for this parameter combination --
    also expected/scientific rather than a bug, but tracked separately by
    callers so the two rejection reasons show up distinctly in the summary.
    """
    from pycbc.filter import sigma
    from pycbc.types import FrequencySeries
    from pycbc.waveform import get_td_waveform
    import pycbc.psd

    try:
        hp, hc = get_td_waveform(
            approximant=config.CONFIG["waveform_approximant"],
            mass1=params["mass_1"], mass2=params["mass_2"],
            spin1z=params["chi_1"], spin2z=params["chi_2"],
            distance=params["distance"],
            delta_t=1.0 / sample_rate,
            f_lower=f_lower,
        )
    except (RuntimeError, ValueError) as exc:
        logger.debug("Waveform generation failed for params=%s: %s", params, exc)
        raise WaveformGenerationError(str(exc)) from exc

    waveform = hp.numpy()
    peak_idx = int(np.argmax(np.abs(waveform)))

    freqs, psd_vals = estimate_psd(noise_seg, sample_rate)
    delta_f = freqs[1] - freqs[0]
    psd_estimate = pycbc.psd.interpolate(FrequencySeries(psd_vals, delta_f=delta_f), hp.delta_f)
    optimal_snr = sigma(hp, psd=psd_estimate, low_frequency_cutoff=f_lower)

    if not np.isfinite(optimal_snr):
        raise ValueError(f"Matched-filter SNR is not finite ({optimal_snr}) for params={params}.")

    if optimal_snr < snr_threshold:
        logger.debug("Injection rejected: SNR=%.2f < threshold=%.1f (params=%s).",
                     optimal_snr, snr_threshold, params)
        return None
    logger.debug("Injection accepted: SNR=%.2f.", optimal_snr)

    window = _compute_injection_window(peak_idx, len(waveform), merger_offset_sec, sample_rate,
                                        buffer_sec, safety_margin_sec, len(noise_seg))

    injected_buffer = noise_seg[window["buf_start"]:window["buf_end"]].copy()
    local_start = window["local_start"]
    injected_buffer[local_start:local_start + len(waveform)] += waveform
    validate_finite(injected_buffer, "injected_buffer")

    return dict(strain=injected_buffer, params=params, snr=float(optimal_snr),
                merger_idx=window["local_merger_idx"])


def validate_injection_dataset(results, expected_n, noise_keys, snr_threshold):
    if not isinstance(results, list) or len(results) != expected_n:
        raise ValueError(f"Expected {expected_n} injections, got "
                         f"{len(results) if isinstance(results, list) else type(results)}.")
    for i, r in enumerate(results):
        validate_strain(r["strain"], f"injection[{i}].strain")
        if r["noise_key"] not in noise_keys:
            raise ValueError(f"injection[{i}] references unknown noise_key {r['noise_key']}.")
        if not np.isfinite(r["snr"]) or r["snr"] < snr_threshold:
            raise ValueError(f"injection[{i}] has invalid SNR={r['snr']} (threshold={snr_threshold}).")


def generate_injection_dataset(run_name: str, noise_segments_dict: dict, seed: int, n_target: int):
    rng = np.random.default_rng(seed)
    noise_keys = list(noise_segments_dict.keys())
    sample_rate = config.CONFIG["sample_rate_hz"]
    merger_lo, _ = config.CONFIG["merger_offset_bounds_sec"]

    results = []
    n_attempted = n_rejected_snr = n_failed_waveform = 0
    log_every = 100

    while len(results) < n_target:
        n_attempted += 1
        noise_key = rng.choice(noise_keys)
        noise_seg, _ = noise_segments_dict[noise_key]

        merger_offset_sec = rng.uniform(merger_lo, noise_seg.shape[0] / sample_rate - merger_lo)
        params = draw_bbh_params(rng)

        try:
            result = generate_injection(
                params, noise_seg, sample_rate=sample_rate, merger_offset_sec=merger_offset_sec,
                f_lower=config.CONFIG["f_lower_hz"], snr_threshold=config.CONFIG["snr_threshold"],
                buffer_sec=config.CONFIG["injection_buffer_sec"],
                safety_margin_sec=config.CONFIG["injection_safety_margin_sec"],
            )
        except WaveformGenerationError:
            n_failed_waveform += 1
            continue

        if result is None:
            n_rejected_snr += 1
            continue

        result["noise_key"] = noise_key
        results.append(result)

        if len(results) % log_every == 0:
            logger.info("%s: %d/%d accepted injections generated (%d attempted so far).",
                        run_name, len(results), n_target, n_attempted)

    logger.info(
        "%s injection generation complete. Accepted: %d. Rejected below SNR threshold: %d. "
        "Failed waveform generation: %d. Total attempted: %d (%.1f%% overall rejection rate).",
        run_name, len(results), n_rejected_snr, n_failed_waveform, n_attempted,
        100 * (n_attempted - len(results)) / n_attempted,
    )
    return results
