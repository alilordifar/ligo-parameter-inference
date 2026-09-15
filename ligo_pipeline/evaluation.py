"""Calibration evaluation: Simulation-Based Calibration (SBC), KS-tests
against uniform rank statistics, repeated-subsample comparisons, and a
lightweight PSD-based drift-detection monitoring check.
"""
import numpy as np
from scipy.stats import kstest, ks_2samp, uniform

from .utils import logger, validate_finite


def validate_sbc_ranks(ranks, n_examples_expected: int, n_params_expected: int):
    arr = np.array(ranks.tolist() if hasattr(ranks, "tolist") else ranks, dtype=float)
    if arr.shape != (n_examples_expected, n_params_expected):
        raise ValueError(f"SBC ranks shape {arr.shape}, expected ({n_examples_expected}, {n_params_expected}).")
    validate_finite(arr, "sbc_ranks")


def summarize_sbc_ks(ranks, num_posterior_samples: int, param_names: list, label: str) -> dict:
    print(f"{label} SBC rank KS-test vs. uniform, per parameter:")
    summary = {}
    for i, name in enumerate(param_names):
        normalized_ranks = np.array(ranks[:, i].tolist()) / num_posterior_samples
        stat, pval = kstest(normalized_ranks, uniform.cdf)
        summary[name] = {"ks_stat": float(stat), "p_value": float(pval)}
        print(f"  {name}: KS stat={stat:.3f}, p-value={pval:.3f}")
    logger.info("%s SBC KS summary: %s", label,
                {k: round(v["ks_stat"], 3) for k, v in summary.items()})
    return summary


def run_sbc_with_retries(posterior_obj, X_test_t, theta_test_t, num_posterior_samples: int,
                          max_sampling_time_sec: float, max_retries: int, run_name: str):
    """Per-example SBC bounded by a sampling-time budget and a retry count.
    A handful of examples can cause the posterior's rejection-sampling
    acceptance rate to collapse toward zero -- rather than hanging, each
    example gets `max_retries` attempts of up to `max_sampling_time_sec`
    each (retries can succeed because rejection sampling is stochastic), and
    is recorded in `failed_indices` -- NOT silently zero-filled -- if none
    succeed."""
    import torch

    n_test = X_test_t.shape[0]
    n_params = theta_test_t.shape[1]
    ranks = torch.full((n_test, n_params), float("nan"))
    failed_indices = []
    log_every = 50

    for i in range(n_test):
        x = X_test_t[i:i + 1]
        true_theta = theta_test_t[i]
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                samples = posterior_obj.sample(
                    (num_posterior_samples,), x=x, show_progress_bars=False,
                    max_sampling_batch_size=num_posterior_samples,
                    max_sampling_time=max_sampling_time_sec,
                )
                for p in range(n_params):
                    ranks[i, p] = float((samples[:, p] < true_theta[p]).sum())
                success = True
                break
            except Exception as exc:
                logger.debug("%s SBC example %d attempt %d/%d failed: %s",
                            run_name, i, attempt, max_retries, exc)
        if not success:
            failed_indices.append(i)
            logger.warning("%s SBC example %d: posterior sampling failed after %d attempt(s) "
                           "(pathological proposal acceptance) -- excluded from calibration stats.",
                           run_name, i, max_retries)
        if (i + 1) % log_every == 0:
            logger.info("%s SBC: %d/%d examples processed (%d failed so far).",
                        run_name, i + 1, n_test, len(failed_indices))

    logger.info("%s SBC complete: %d/%d succeeded, %d failed: %s",
                run_name, n_test - len(failed_indices), n_test, len(failed_indices), failed_indices)
    return ranks, failed_indices


def clean_sbc_ranks(ranks, failed_indices: list, label: str):
    import torch

    valid_mask = torch.ones(ranks.shape[0], dtype=torch.bool)
    if failed_indices:
        valid_mask[failed_indices] = False
    clean = ranks[valid_mask]
    validate_finite(clean.tolist(), f"{label}_clean_ranks")
    print(f"using {clean.shape[0]} valid {label} examples (excluded {len(failed_indices)}: {failed_indices})")
    return clean


def repeated_subsample_ks(ranks_clean, num_posterior_samples: int, param_names: list,
                           subsample_n: int, n_repeats: int, base_seed: int) -> dict:
    """O4a has far more valid examples than O3a's held-out set, so a single
    KS statistic on the full O4a set isn't an apples-to-apples comparison.
    Instead, repeatedly subsample O4a down to O3a's held-out size and report
    the range -- if O3a's single value falls inside that range, the two runs
    aren't distinguishable at this sample size."""
    import torch

    results_by_param = {name: [] for name in param_names}
    for seed in range(base_seed, base_seed + n_repeats):
        rng_s = np.random.default_rng(seed)
        idx = rng_s.choice(ranks_clean.shape[0], size=subsample_n, replace=False)
        idx_t = torch.tensor(idx.tolist(), dtype=torch.long)
        sub = ranks_clean[idx_t]
        for i, name in enumerate(param_names):
            normalized = np.array(sub[:, i].tolist()) / num_posterior_samples
            stat, _ = kstest(normalized, uniform.cdf)
            results_by_param[name].append(stat)
    return results_by_param


# ---------------------------------------------------------------------------
# Lightweight PSD-based drift-detection monitoring check
# ---------------------------------------------------------------------------

def detect_psd_drift(freqs_a, psd_a, freqs_b, psd_b, band_low: float, band_high: float,
                      ks_alpha: float = 0.05, relative_threshold: float = 0.20) -> dict:
    """Cheap, pre-SBC monitoring signal: would comparing two runs' average
    noise ASD curves in the analysis band alone have flagged "retraining
    warranted" before ever running the (expensive) SBC evaluation?

    Reuses `average_psd` output already computed for the O3a-vs-O4a ASD
    plot -- no new model training, no new data acquisition. Two independent,
    complementary criteria, either of which flags drift:

    1. A two-sample KS-test treating each run's in-band ASD samples (across
       frequency bins) as an empirical distribution -- tests whether the two
       curves' *shape* in-band differs by more than sampling noise would
       explain. p-value below `ks_alpha` flags drift.
    2. The median relative ASD difference in-band (interpolated onto a
       common frequency grid) -- a simple, interpretable magnitude of
       "how much louder/quieter is the new run's noise floor." Above
       `relative_threshold` flags drift.

    This is a monitoring-signal demonstration, not a validated drift
    detector -- see TUTORIAL.md and PROPOSAL.md for the caveat that these
    thresholds are illustrative defaults, not tuned/calibrated values.
    """
    freqs_a, psd_a = np.asarray(freqs_a), np.asarray(psd_a)
    freqs_b, psd_b = np.asarray(freqs_b), np.asarray(psd_b)

    mask_a = (freqs_a >= band_low) & (freqs_a <= band_high)
    mask_b = (freqs_b >= band_low) & (freqs_b <= band_high)
    asd_a_band = np.sqrt(psd_a[mask_a])
    asd_b_band = np.sqrt(psd_b[mask_b])

    ks_stat, ks_pvalue = ks_2samp(asd_a_band, asd_b_band)
    ks_flag = bool(ks_pvalue < ks_alpha)

    # common grid for a magnitude-of-difference comparison (independent of
    # the two runs' PSDs having different frequency bin counts)
    common_freqs = freqs_a[mask_a]
    asd_b_interp = np.interp(common_freqs, freqs_b, np.sqrt(psd_b))
    relative_diff = np.abs(asd_b_interp - np.sqrt(psd_a[mask_a])) / np.sqrt(psd_a[mask_a])
    median_relative_diff = float(np.median(relative_diff))
    magnitude_flag = bool(median_relative_diff > relative_threshold)

    flagged = ks_flag or magnitude_flag
    result = {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "ks_flag": ks_flag,
        "median_relative_asd_diff": median_relative_diff,
        "magnitude_flag": magnitude_flag,
        "flagged_retrain_warranted": flagged,
    }
    logger.info("PSD drift check (%.0f-%.0f Hz band): KS stat=%.3f p=%.3g (flag=%s), "
                "median relative ASD diff=%.1f%% (flag=%s) -> retrain-warranted=%s",
                band_low, band_high, ks_stat, ks_pvalue, ks_flag,
                100 * median_relative_diff, magnitude_flag, flagged)
    return result
