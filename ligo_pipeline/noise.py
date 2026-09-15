"""GWOSC noise-segment acquisition and quality vetting, for both the
training run (O3a) and the cross-run evaluation set (O4a).

`fetch_clean_noise` distinguishes two very different outcomes:
- **expected scientific rejection** (implausible noise std, failing the DQ
  bitmask, spanning multiple frame files, NaNs) -- logged at DEBUG and the
  caller just moves on to the next candidate;
- **unexpected failure** (malformed HDF5, wrong sample rate) -- logged at
  ERROR/raised, because that indicates something is actually broken rather
  than "this particular second of data isn't usable."
"""
import io

import numpy as np
import requests

import config
from .preprocessing import estimate_psd
from .utils import logger, retry, validate_sample_rate, validate_strain

# gwosc/h5py are imported lazily inside the functions that need them (rather
# than at module level) so this module -- and its pure logic, like
# validate_noise_segments/average_psd -- stays importable and unit-testable
# without those (heavier, network-service-specific) packages installed.


@retry(times=3, exceptions=(requests.exceptions.RequestException, ConnectionError, TimeoutError))
def lookup_event_gps(name: str) -> float:
    """GWOSC catalog GPS-time lookup for a named event. Used both for EDA
    sanity checks (a specific known event) and inside noise acquisition
    (excluding a buffer around every cataloged event in a run)."""
    from gwosc.datasets import event_gps
    return event_gps(name)


def validate_noise_segments(segments, expected_duration_sec, expected_sample_rate_hz):
    if not isinstance(segments, dict) or len(segments) == 0:
        raise ValueError("Noise-segment checkpoint is empty or not a dict.")
    expected_len = int(expected_duration_sec * expected_sample_rate_hz)
    for t0, (seg, seg_times) in segments.items():
        validate_strain(seg, f"noise segment @ GPS {t0}", expected_len=expected_len)
        if len(seg_times) != len(seg):
            raise ValueError(f"noise segment @ GPS {t0}: times/strain length mismatch "
                             f"({len(seg_times)} vs {len(seg)}).")


def find_clean_segment_starts(run_name: str, start: float, end: float, segment_duration: float,
                               buffer_sec: float, n_candidates: int):
    """Live GWOSC catalog query -- only ever called when there's no already-
    valid noise-segment checkpoint for this run."""
    from gwosc import datasets

    logger.info("Querying GWOSC event catalog for %s (GPS %.0f-%.0f)...", run_name, start, end)
    try:
        raw_events = datasets.find_datasets(type="events", segment=(start, end))
    except Exception as exc:
        logger.error("Failed to query GWOSC event catalog for %s: %s", run_name, exc)
        raise

    unique_events = sorted(set(e.split("-v")[0] for e in raw_events))
    event_gps_times, n_failed = {}, 0
    for name in unique_events:
        try:
            event_gps_times[name] = lookup_event_gps(name)
        except Exception as exc:
            logger.debug("Could not resolve GPS time for cataloged event %s: %s", name, exc)
            n_failed += 1
    logger.info("%s: %d cataloged events, resolved %d GPS times (%d lookups failed).",
                run_name, len(unique_events), len(event_gps_times), n_failed)

    event_times = np.array(sorted(event_gps_times.values()))

    def is_clean(t):
        nearby = event_times[(event_times > t - buffer_sec) & (event_times < t + segment_duration + buffer_sec)]
        return len(nearby) == 0

    candidate_starts = np.linspace(start, end - segment_duration, n_candidates)
    clean_starts = [t for t in candidate_starts if is_clean(t)]
    logger.info("%s: %d/%d candidate windows are event-free (buffer=%.0fs).",
                run_name, len(clean_starts), n_candidates, buffer_sec)
    return clean_starts


@retry(times=3, exceptions=(requests.exceptions.RequestException, ConnectionError, OSError))
def _download_frame(url: str) -> bytes:
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return resp.content


def fetch_clean_noise(detector: str, start: float, duration: float, sample_rate: float,
                       cbc_bits: tuple, std_range: tuple):
    """Attempt to fetch and quality-vet one noise segment. Returns
    (segment, times, True) on acceptance, else (None, None, False) with the
    reason logged at DEBUG -- rejections here are an expected part of the
    scientific selection process, not errors."""
    from gwosc.locate import get_urls

    try:
        urls = get_urls(detector, start, start + duration, sample_rate=sample_rate)
    except ValueError as exc:
        logger.debug("No GWOSC frame available for %s start=%.0f: %s", detector, start, exc)
        return None, None, False

    if len(urls) != 1:
        logger.debug("Candidate start=%.0f spans %d frame files -- skipping (single-file only).",
                     start, len(urls))
        return None, None, False

    try:
        content = _download_frame(urls[0])
    except requests.exceptions.RequestException as exc:
        logger.warning("Download failed after retries for %s (start=%.0f): %s -- skipping candidate.",
                       urls[0], start, exc)
        return None, None, False

    import h5py

    try:
        with h5py.File(io.BytesIO(content), "r") as f:
            strain = f["strain"]["Strain"][:]
            file_gps_start = f["meta"]["GPSstart"][()]
            file_sample_rate = 1.0 / f["strain"]["Strain"].attrs["Xspacing"]
            dq = f["quality"]["simple"]["DQmask"][:]
    except (OSError, KeyError) as exc:
        logger.error("Malformed GWOSC frame file at %s (start=%.0f): %s -- skipping.", urls[0], start, exc)
        return None, None, False

    validate_sample_rate(file_sample_rate, sample_rate, "GWOSC frame sample_rate")

    start_idx = int((start - file_gps_start) * file_sample_rate)
    end_idx = start_idx + int(duration * file_sample_rate)
    segment = strain[start_idx:end_idx]

    if np.any(np.isnan(segment)):
        logger.debug("Rejected start=%.0f: contains NaNs.", start)
        return None, None, False

    seg_std = segment.std()
    if not (std_range[0] < seg_std < std_range[1]):
        logger.debug("Rejected start=%.0f: implausible broadband std=%.3e (expected within %s).",
                     start, seg_std, std_range)
        return None, None, False

    dq_start = int(start - file_gps_start)
    dq_window = dq[dq_start:dq_start + int(duration)]
    cbc_mask = sum(1 << b for b in cbc_bits)
    if not np.all((dq_window & cbc_mask) == cbc_mask):
        logger.debug("Rejected start=%.0f: fails DQ CBC quality bitmask.", start)
        return None, None, False

    times = start + np.arange(len(segment)) / file_sample_rate
    return segment, times, True


def acquire_noise_segments(run_name: str, start: float, end: float, target_n: int,
                            segment_duration: float, buffer_sec: float, seed: int):
    n_candidates = target_n * 15  # oversample; DQ/std rejection typically removes a large fraction
    clean_starts = find_clean_segment_starts(run_name, start, end, segment_duration, buffer_sec, n_candidates)

    rng = np.random.default_rng(seed)
    ordered_starts = [clean_starts[i] for i in rng.permutation(len(clean_starts))]

    segments, n_attempted = {}, 0
    for t0 in ordered_starts:
        if len(segments) >= target_n:
            break
        n_attempted += 1
        seg, seg_times, ok = fetch_clean_noise(config.CONFIG["detector"], t0, segment_duration,
                                                config.CONFIG["sample_rate_hz"], config.CONFIG["dq_cbc_bits"],
                                                config.CONFIG["noise_std_range"])
        if ok:
            segments[t0] = (seg, seg_times)
            logger.info("%s: accepted noise segment start=%.0f, std=%.3e (%d/%d).",
                        run_name, t0, seg.std(), len(segments), target_n)

    n_rejected = n_attempted - len(segments)
    logger.info("%s noise acquisition done: %d/%d accepted, %d candidates attempted, %d rejected.",
                run_name, len(segments), target_n, n_attempted, n_rejected)
    if len(segments) < target_n:
        logger.error("%s: only found %d/%d clean noise segments after exhausting %d candidates.",
                     run_name, len(segments), target_n, len(clean_starts))
        raise RuntimeError(f"{run_name}: insufficient clean noise segments ({len(segments)}/{target_n}). "
                            "Widen the candidate pool or relax the DQ/std filters.")
    return segments


def average_psd(segments_dict, sample_rate):
    """Mean Welch PSD across a run's quality-vetted noise segments -- used
    both for the O3a-vs-O4a ASD comparison plot and as the input to the
    lightweight PSD-drift monitoring check (see ligo_pipeline.evaluation)."""
    psds = []
    freqs = None
    for t0, (seg, seg_times) in segments_dict.items():
        freqs, psd = estimate_psd(seg, sample_rate)
        psds.append(psd)
    return freqs, np.mean(psds, axis=0)
