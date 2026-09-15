"""Infrastructure helpers shared across the pipeline: atomic checkpoint
writes with provenance metadata, input validation, bounded network retries,
and logging setup.

Deliberately config-agnostic (no `import config` here) so it stays reusable
on its own -- callers pass in whatever metadata/paths they need.
"""
import functools
import hashlib
import io
import json
import logging
import os
import pickle
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LOGGER_NAME = "ligo_pipeline"
logger = logging.getLogger(LOGGER_NAME)


def setup_logging(log_dir: Path, level_console: int = logging.INFO,
                   level_file: int = logging.DEBUG) -> logging.Logger:
    """Configure the pipeline's logger once. Safe to call more than once
    (e.g. on notebook re-run) -- clears any handlers it previously attached
    instead of stacking duplicate handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"

    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)

    import sys
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level_console)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    log.addHandler(console)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s:%(lineno)d %(message)s"))
    log.addHandler(file_handler)

    log.info("Logging initialized. Console=%s, full debug log=%s", logging.getLevelName(level_console), log_path)
    return log


# ---------------------------------------------------------------------------
# Atomic checkpoint writes + provenance-checked reuse
# ---------------------------------------------------------------------------

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def config_fingerprint(config_dict: dict, *keys: str) -> str:
    """Short hash of selected keys from a config dict -- lets a metadata
    sidecar record 'which parameters this artifact depends on' compactly."""
    subset = {k: config_dict[k] for k in keys}
    blob = json.dumps(subset, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def save_checkpoint(path: Path, obj, metadata: dict, overwrite: bool = False) -> None:
    """Atomically pickle `obj` to `path` plus a JSON metadata sidecar
    (`<path>.meta.json`) recording how it was produced. Refuses to clobber an
    existing artifact unless `overwrite=True` is passed explicitly."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing artifact {path} (overwrite=False). "
            "This is a safety guard against accidentally destroying a completed "
            "result -- pass overwrite=True if that is genuinely what you want."
        )
    full_meta = {"created_utc": datetime.now(timezone.utc).isoformat(), **metadata}
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(path, buf.getvalue())
    _atomic_write_bytes(_meta_path(path), json.dumps(full_meta, indent=2, default=str).encode())
    logger.info("Saved checkpoint %s (%.2f MB) + metadata sidecar.", path, path.stat().st_size / 1e6)


def load_checkpoint(path: Path, expected_meta: dict | None = None, validator=None):
    """Load a checkpoint if it exists and is compatible with `expected_meta`.

    Returns (obj, status) with status one of:
      "missing"      -- no file at all
      "corrupt"      -- file exists but failed to unpickle, or failed
                         structural validation
      "incompatible" -- file exists and loads fine, but its metadata sidecar
                         disagrees with the current configuration
      "legacy"       -- file exists, loads fine, structurally valid, but has
                         no metadata sidecar to verify provenance against
      "reused"       -- file exists, loads fine, and is confirmed compatible

    In every non-"reused" case the caller should regenerate the artifact;
    this function never returns data it cannot vouch for.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None, "missing"

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except (EOFError, pickle.UnpicklingError, AttributeError, ModuleNotFoundError) as exc:
        logger.error("Checkpoint %s exists but failed to load (%s: %s) -- treating as missing "
                     "and regenerating.", path, type(exc).__name__, exc)
        return None, "corrupt"

    status = "reused"
    meta_path = _meta_path(path)
    if meta_path.exists():
        try:
            stored_meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Metadata sidecar %s is corrupt (%s) -- cannot verify provenance of %s, "
                           "regenerating to be safe.", meta_path, exc, path)
            return None, "incompatible"

        if expected_meta:
            mismatches = {
                k: {"cached": stored_meta.get(k), "current": v}
                for k, v in expected_meta.items()
                if k in stored_meta and stored_meta[k] != v
            }
            if mismatches:
                logger.warning("Checkpoint %s was produced by a different configuration than the "
                               "current run and will NOT be reused: %s", path, mismatches)
                return None, "incompatible"
    else:
        logger.warning("Checkpoint %s has no metadata sidecar (legacy artifact from before this "
                       "project's provenance tracking existed) -- provenance cannot be verified, "
                       "falling back to structural validation only.", path)
        status = "legacy"

    if validator is not None:
        try:
            validator(obj)
        except Exception as exc:
            logger.error("Checkpoint %s failed structural validation (%s: %s) -- regenerating.",
                         path, type(exc).__name__, exc)
            return None, "corrupt"

    logger.info("Reusing existing checkpoint %s [%s] -- skipping regeneration.", path, status)
    return obj, status


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_finite(arr, name: str = "array") -> None:
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(f"{name} contains {n_bad} NaN/infinite value(s) out of {arr.size} (shape={arr.shape}).")


def validate_strain(strain, name: str = "strain", expected_len: int | None = None) -> None:
    strain = np.asarray(strain)
    if strain.size == 0:
        raise ValueError(f"{name} is empty.")
    validate_finite(strain, name)
    if expected_len is not None and strain.shape[0] != expected_len:
        raise ValueError(f"{name} has length {strain.shape[0]}, expected {expected_len}.")


def validate_psd(freqs, psd, name: str = "psd") -> None:
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    if freqs.shape != psd.shape:
        raise ValueError(f"{name}: freqs shape {freqs.shape} != psd shape {psd.shape}.")
    validate_finite(psd, name)
    if np.any(psd < 0):
        raise ValueError(f"{name} contains {int(np.sum(psd < 0))} negative power value(s).")


def validate_sample_rate(actual: float, expected: float, name: str = "sample_rate") -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: got {actual} Hz, expected {expected} Hz.")


# ---------------------------------------------------------------------------
# Bounded retry (network calls only)
# ---------------------------------------------------------------------------

def retry(times: int = 3, exceptions=(ConnectionError, OSError), backoff_sec: float = 2.0):
    """Bounded retry for network calls only, with linear backoff. Anything
    outside `exceptions` (e.g. a ValueError from bad scientific input)
    propagates immediately on the first attempt -- retrying a deterministic
    error just wastes `times` attempts' worth of time."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    logger.warning("Attempt %d/%d failed for %s(): %s", attempt, times, fn.__name__, exc)
                    if attempt < times:
                        time.sleep(backoff_sec * attempt)
            logger.error("All %d attempts failed for %s() -- giving up.", times, fn.__name__)
            raise last_exc
        return wrapper
    return decorator
