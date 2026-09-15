"""Signal-processing primitives (PSD estimation, whitening, bandpass) and
the per-injection preprocessing (whiten + bandpass + crop) applied before
training/evaluation. See TUTORIAL.md for the math behind each step.
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt

import config
from .utils import validate_finite, validate_psd, validate_strain, logger


def estimate_psd(strain, sample_rate, nperseg_sec: int | None = None):
    """Welch PSD estimate. Used both for diagnostic whitening (EDA) and for
    matched-filter SNR computation (injections)."""
    nperseg_sec = nperseg_sec if nperseg_sec is not None else config.CONFIG["psd_nperseg_sec"]
    freqs, psd = signal.welch(strain, fs=sample_rate, nperseg=sample_rate * nperseg_sec)
    validate_psd(freqs, psd)
    return freqs, psd


def whiten(strain, sample_rate, psd_freqs, psd_vals):
    """Frequency-domain whitening: divide by sqrt(PSD) interpolated onto the
    strain's own FFT frequency bins."""
    validate_strain(strain, "strain_to_whiten")
    fft_data = np.fft.rfft(strain)
    fft_freqs = np.fft.rfftfreq(len(strain), d=1.0 / sample_rate)
    psd_interp = np.interp(fft_freqs, psd_freqs, psd_vals)
    whitened = np.fft.irfft(fft_data / np.sqrt(psd_interp), n=len(strain))
    validate_finite(whitened, "whitened_strain")
    return whitened


def bandpass(strain, sample_rate, low: float | None = None, high: float | None = None,
             order: int | None = None):
    low = low if low is not None else config.CONFIG["bandpass_low_hz"]
    high = high if high is not None else config.CONFIG["bandpass_high_hz"]
    order = order if order is not None else config.CONFIG["bandpass_order"]
    nyq = sample_rate / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    out = filtfilt(b, a, strain)
    validate_finite(out, "bandpassed_strain")
    return out


def whiten_and_bandpass(strain, sample_rate, psd_freqs, psd_vals):
    """The whiten -> bandpass step used repeatedly throughout the pipeline."""
    return bandpass(whiten(strain, sample_rate, psd_freqs, psd_vals), sample_rate)


def compute_derived_params(params: dict) -> np.ndarray:
    """Chirp mass, mass ratio, effective spin, distance -- the 4 target
    parameters, derived from the raw draw. See TUTORIAL.md for the formulas
    and why these (rather than raw component masses/spins) are the natural
    inference targets."""
    m1, m2 = params["mass_1"], params["mass_2"]
    chi1, chi2 = params["chi_1"], params["chi_2"]

    chirp_mass = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    mass_ratio = m2 / m1  # m1 >= m2 already enforced in draw_bbh_params
    chi_eff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    distance = params["distance"]

    return np.array([chirp_mass, mass_ratio, chi_eff, distance])


def preprocess_injection(result: dict, noise_segments_dict: dict, window_sec: float, sample_rate: float):
    """Whiten (using the ORIGINAL noise segment's own PSD, not the injected
    array -- matches real practice: PSD estimated from off-source/background
    noise) + bandpass + crop around merger."""
    strain = result["strain"]
    merger_idx = result["merger_idx"]
    noise_seg, _ = noise_segments_dict[result["noise_key"]]

    freqs, psd_vals = estimate_psd(noise_seg, sample_rate)
    bandpassed = whiten_and_bandpass(strain, sample_rate, freqs, psd_vals)

    start = merger_idx - int(window_sec * sample_rate)
    end = merger_idx + int(window_sec * sample_rate)
    cropped = bandpassed[start:end]
    validate_strain(cropped, "preprocessed_injection", expected_len=2 * int(window_sec * sample_rate))
    return cropped


def validate_training_data(td: dict, n_train: int, n_holdout_o3a: int, n_o4a: int, n_params: int):
    for key in ("X_train", "theta_train", "X_test_o3a", "theta_test_o3a", "X_test_o4a", "theta_test_o4a"):
        if key not in td:
            raise ValueError(f"training_data checkpoint missing key '{key}'.")
        validate_finite(td[key], key)

    if td["X_train"].shape[0] != n_train:
        raise ValueError(f"X_train has {td['X_train'].shape[0]} rows, expected {n_train}.")
    if td["X_test_o3a"].shape[0] != n_holdout_o3a:
        raise ValueError(f"X_test_o3a has {td['X_test_o3a'].shape[0]} rows, expected {n_holdout_o3a}.")
    if td["X_test_o4a"].shape[0] != n_o4a:
        raise ValueError(f"X_test_o4a has {td['X_test_o4a'].shape[0]} rows, expected {n_o4a}.")
    for theta_key in ("theta_train", "theta_test_o3a", "theta_test_o4a"):
        if td[theta_key].shape[1] != n_params:
            raise ValueError(f"{theta_key} has {td[theta_key].shape[1]} parameter columns, expected {n_params}.")
