import json
import pickle

import numpy as np
import pytest

from ligo_pipeline.utils import (
    config_fingerprint,
    load_checkpoint,
    retry,
    save_checkpoint,
    validate_finite,
    validate_psd,
    validate_sample_rate,
    validate_strain,
)


# --- checkpoint round trip ---------------------------------------------

def test_save_then_load_checkpoint_is_reused(tmp_path):
    path = tmp_path / "artifact.pkl"
    obj = {"x": np.array([1.0, 2.0, 3.0])}
    save_checkpoint(path, obj, metadata={"seed": 1})

    loaded, status = load_checkpoint(path)
    assert status == "reused"
    np.testing.assert_array_equal(loaded["x"], obj["x"])


def test_save_checkpoint_refuses_overwrite_by_default(tmp_path):
    path = tmp_path / "artifact.pkl"
    save_checkpoint(path, {"x": 1}, metadata={})
    with pytest.raises(FileExistsError):
        save_checkpoint(path, {"x": 2}, metadata={})
    # overwrite=True is allowed
    save_checkpoint(path, {"x": 2}, metadata={}, overwrite=True)
    loaded, _ = load_checkpoint(path)
    assert loaded["x"] == 2


def test_load_checkpoint_missing_file(tmp_path):
    obj, status = load_checkpoint(tmp_path / "does_not_exist.pkl")
    assert obj is None
    assert status == "missing"


def test_load_checkpoint_corrupt_pickle(tmp_path):
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"not a valid pickle stream")
    obj, status = load_checkpoint(path)
    assert obj is None
    assert status == "corrupt"


def test_load_checkpoint_incompatible_metadata(tmp_path):
    path = tmp_path / "artifact.pkl"
    save_checkpoint(path, {"x": 1}, metadata={"seed": 1, "n": 100})

    obj, status = load_checkpoint(path, expected_meta={"seed": 2})
    assert obj is None
    assert status == "incompatible"

    # unchanged metadata is still reused
    obj, status = load_checkpoint(path, expected_meta={"seed": 1})
    assert status == "reused"
    assert obj["x"] == 1


def test_load_checkpoint_legacy_missing_sidecar(tmp_path):
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump({"x": 1}, f)
    obj, status = load_checkpoint(path)
    assert status == "legacy"
    assert obj["x"] == 1


def test_load_checkpoint_runs_validator_and_flags_corrupt_on_failure(tmp_path):
    path = tmp_path / "artifact.pkl"
    save_checkpoint(path, {"x": -1}, metadata={})

    def validator(obj):
        if obj["x"] < 0:
            raise ValueError("x must be non-negative")

    obj, status = load_checkpoint(path, validator=validator)
    assert obj is None
    assert status == "corrupt"


def test_config_fingerprint_deterministic_and_sensitive_to_values():
    cfg = {"a": 1, "b": 2, "c": 3}
    fp1 = config_fingerprint(cfg, "a", "b")
    fp2 = config_fingerprint(cfg, "a", "b")
    assert fp1 == fp2

    fp3 = config_fingerprint({"a": 1, "b": 999}, "a", "b")
    assert fp1 != fp3


# --- validation -----------------------------------------------------------

def test_validate_finite_rejects_nan():
    with pytest.raises(ValueError):
        validate_finite(np.array([1.0, np.nan, 3.0]))


def test_validate_finite_accepts_finite():
    validate_finite(np.array([1.0, 2.0, 3.0]))  # no raise


def test_validate_strain_rejects_empty():
    with pytest.raises(ValueError):
        validate_strain(np.array([]))


def test_validate_strain_rejects_wrong_length():
    with pytest.raises(ValueError):
        validate_strain(np.zeros(10), expected_len=20)


def test_validate_psd_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        validate_psd(np.zeros(10), np.zeros(5))


def test_validate_psd_rejects_negative_power():
    freqs = np.arange(5.0)
    psd = np.array([1.0, 2.0, -1.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        validate_psd(freqs, psd)


def test_validate_sample_rate_mismatch():
    with pytest.raises(ValueError):
        validate_sample_rate(2048, 4096)
    validate_sample_rate(4096, 4096)  # no raise


# --- retry ------------------------------------------------------------

def test_retry_retries_on_matching_exception_then_succeeds():
    calls = {"n": 0}

    @retry(times=3, exceptions=(ConnectionError,), backoff_sec=0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    @retry(times=2, exceptions=(ConnectionError,), backoff_sec=0)
    def always_fails():
        calls["n"] += 1
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError):
        always_fails()
    assert calls["n"] == 2


def test_retry_does_not_retry_unrelated_exceptions():
    calls = {"n": 0}

    @retry(times=3, exceptions=(ConnectionError,), backoff_sec=0)
    def bad_input():
        calls["n"] += 1
        raise ValueError("not a network error")

    with pytest.raises(ValueError):
        bad_input()
    assert calls["n"] == 1
