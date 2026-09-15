"""Single source of truth for every tunable parameter in the pipeline.

Nothing in ligo_pipeline/ or notebooks/main.ipynb should hardcode a value
that also appears here -- filenames, metadata sidecars, and log lines all
read directly from CONFIG, so the logged record of "what ran" can never
drift from what actually ran.
"""
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ARTIFACT_DIRS = {
    "noise": ARTIFACTS_DIR / "data" / "noise",
    "injections": ARTIFACTS_DIR / "data" / "injections",
    "training_data": ARTIFACTS_DIR / "data" / "training_data",
    "models": ARTIFACTS_DIR / "models",
    "sbc": ARTIFACTS_DIR / "results" / "sbc",
    "audio": ARTIFACTS_DIR / "audio",
    "logs": ARTIFACTS_DIR / "logs",
}


CONFIG = {
    # --- raw ingested-strain EDA (bronze S3 bucket) ---
    "s3_bucket": "signal-platform-dev-471112934830",
    "s3_region": "us-east-1",
    "eda_event_name": "GW230902_150325",
    "eda_prefix": "raw/domain=ligo/source_id=H1/year=2023/month=09/day=02/",
    "eda_json_key": "raw/domain=ligo/source_id=H1/year=2023/month=09/day=02/H1_1693666670.000_4096.000.json",
    "eda_npy_key": "raw/domain=ligo/source_id=H1/year=2023/month=09/day=02/H1_1693666670.000_4096.000.npy",
    "gw150914_prefix": "raw/domain=ligo/source_id=H1/year=2015/",
    "gw150914_json_key": "raw/domain=ligo/source_id=H1/year=2015/month=09/day=14/H1_1442224245.000_4096.000.json",
    "gw150914_npy_key": "raw/domain=ligo/source_id=H1/year=2015/month=09/day=14/H1_1442224245.000_4096.000.npy",

    # --- detector / observing runs ---
    "detector": "H1",
    "train_run": "O3a",
    "eval_run": "O4a",

    # --- signal processing ---
    "sample_rate_hz": 4096,
    "bandpass_low_hz": 35,
    "bandpass_high_hz": 350,
    "bandpass_order": 4,
    "psd_nperseg_sec": 4,  # Welch nperseg = sample_rate_hz * psd_nperseg_sec

    # --- noise segment acquisition ---
    "n_noise_segments_per_run": 8,
    "noise_segment_duration_sec": 1024,
    "event_exclusion_buffer_sec": 3600,
    "noise_std_range": (1e-19, 1e-17),
    "dq_cbc_bits": (0, 1, 2, 3),  # required-clean DQ bits (CAT1-4 style CBC flags)

    # --- injections ---
    "waveform_approximant": "IMRPhenomD",
    "f_lower_hz": 20.0,
    "mass_prior_msun": (5.0, 100.0),
    "spin_prior": (0.0, 0.8),
    "distance_prior_mpc": (100.0, 2000.0),  # comoving-volume-weighted; truncated from Bilby's 5000 Mpc default
    "snr_threshold": 8.0,
    "n_injections_per_run": 800,
    "merger_offset_bounds_sec": (60.0, None),  # upper bound = segment length - 60s, filled in at draw time
    "injection_buffer_sec": 15.0,
    "injection_safety_margin_sec": 2.0,

    # --- preprocessing / split ---
    "crop_window_sec": 2.0,
    "n_holdout_o3a": 50,

    # --- model ---
    "model_family": "SNPE",
    "density_estimator": "MAF",
    "param_names": ["chirp_mass", "mass_ratio", "chi_eff", "distance"],
    "prior_low": [1.0, 0.05, -0.8, 100.0],
    "prior_high": [90.0, 1.0, 0.8, 2000.0],

    # --- SBC ---
    "num_posterior_samples_o3a": 1000,
    "num_posterior_samples_o4a": 200,
    "sbc_max_sampling_time_sec": 10.0,  # per-example, per-attempt cap; shared by O3a and O4a SBC
    "sbc_max_retries": 2,
    "sbc_subsample_n": 50,
    "sbc_subsample_repeats": 10,
    "sbc_subsample_seed": 99,

    # --- PSD-based drift-detection monitoring check (cheap, pre-SBC signal) ---
    # Compares O3a vs. O4a average ASD curves within the analysis band
    # (bandpass_low_hz-bandpass_high_hz). Two independent criteria, either
    # of which flags "retraining warranted":
    #   1. a two-sample KS-test p-value below psd_drift_ks_alpha, or
    #   2. a median relative ASD difference above psd_drift_relative_threshold.
    # Thresholds are illustrative defaults for a monitoring-signal demo, not
    # tuned/validated values -- see TUTORIAL.md for the caveat.
    "psd_drift_ks_alpha": 0.05,
    "psd_drift_relative_threshold": 0.20,

    # --- reproducibility ---
    "random_seed_o3a_noise": 0,
    "random_seed_o3a_injections": 42,
    "random_seed_o4a_injections": 43,
    "random_seed_train_test_split": 7,
    "random_seed_sbc_subsample_repeats_base": 0,
    "global_seed": 1234,
}


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs. Does NOT guarantee bit-identical
    reruns across different hardware/BLAS/CUDA builds -- only that the same
    machine/library versions will reproduce the same draws."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_software_versions() -> dict:
    """Snapshot of installed library versions, for checkpoint metadata
    sidecars. Called explicitly (not at import time) so config.py stays
    importable -- e.g. by tests -- without every heavy ML/GW dependency
    installed."""
    import platform
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "pandas", "torch", "pycbc", "sbi", "gwosc", "h5py", "boto3"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions
