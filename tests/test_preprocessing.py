"""Unit tests for Phase 1 data loading, preprocessing, and drift utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from secom_causal.drift import compute_psi, wilson_ci
from secom_causal.preprocessing import SecomPreprocessor, time_ordered_split


def _make_toy_dataset(n: int = 20) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(123)
    timestamps = pd.date_range("2008-07-19", periods=n, freq="h")
    X = pd.DataFrame({
        "const_col": np.ones(n),
        "all_nan": np.full(n, np.nan),
        "high_miss": np.where(np.arange(n) < 15, np.nan, rng.normal(size=n)),
        "flag_miss": np.where(np.arange(n) % 3 == 0, np.nan, rng.normal(10.0, 2.0, size=n)),
        "clean_a": rng.normal(0.0, 1.0, size=n),
        "clean_b": rng.normal(5.0, 3.0, size=n),
    })
    # Inject a test-only constant-looking or shifted value to verify no leakage.
    y = pd.Series((np.arange(n) % 5 == 0).astype(int), name="fail")
    return X, y, pd.Series(timestamps, name="timestamp")


def test_time_ordered_split_preserves_chronology() -> None:
    X, y, ts = _make_toy_dataset(20)
    split = time_ordered_split(X, y, ts, train_fraction=0.7)
    assert len(split.X_train) == 14
    assert len(split.X_test) == 6
    assert split.ts_train.max() <= split.ts_test.min()


def test_time_ordered_split_rejects_unsorted_timestamps() -> None:
    X, y, ts = _make_toy_dataset(10)
    ts_unsorted = ts.sample(frac=1.0, random_state=7).reset_index(drop=True)
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        time_ordered_split(X, y, ts_unsorted, train_fraction=0.7)


def test_preprocessor_drops_constants_and_high_missing_without_leakage() -> None:
    X, y, ts = _make_toy_dataset(20)
    # In train (first 14 rows), high_miss is 100% NaN; in test (last 6 rows) it is observed.
    # Also make clean_a have median ~0 on train and +1000 on test to verify train-only fit.
    X.loc[14:, "clean_a"] = 1000.0
    X.loc[15, "clean_a"] = np.nan

    split = time_ordered_split(X, y, ts, train_fraction=0.7)
    prep = SecomPreprocessor(
        drop_constant=True,
        high_missing_flag_threshold=0.30,
        high_missing_drop_threshold=0.55,
        imputation_method="median_indicator",
        add_missing_indicators=True,
        indicator_min_missing_rate=0.05,
    ).fit(split.X_train)

    assert "const_col" in prep.constant_features_
    assert "all_nan" in prep.constant_features_
    assert "flag_miss" in prep.flagged_high_missing_
    assert "flag_miss" in prep.kept_sensor_features_
    assert "miss__flag_miss" in prep.output_features_

    X_tr = prep.transform(split.X_train)
    X_te = prep.transform(split.X_test)

    assert not X_tr.isna().any().any()
    assert not X_te.isna().any().any()
    # Imputed test clean_a[15] must use train median (< 10), not test values (~1000).
    assert X_te.loc[15, "clean_a"] < 10.0


def test_preprocessor_iterative_imputation_runs_cleanly() -> None:
    X, y, ts = _make_toy_dataset(30)
    split = time_ordered_split(X, y, ts, train_fraction=0.7)
    prep = SecomPreprocessor(
        imputation_method="iterative",
        iterative_max_iter=5,
        iterative_n_nearest_features=3,
        random_state=42,
    ).fit(split.X_train)
    X_tr = prep.transform(split.X_train)
    X_te = prep.transform(split.X_test)
    assert not X_tr.isna().any().any()
    assert not X_te.isna().any().any()
    assert list(X_tr.columns) == list(X_te.columns)


def test_psi_and_wilson_ci_bounds() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=500)
    same = rng.normal(0.0, 1.0, size=500)
    shifted = rng.normal(2.0, 1.0, size=500)

    assert compute_psi(ref, same) < 0.1
    assert compute_psi(ref, shifted) > 0.5

    lo, hi = wilson_ci(10, 100)
    assert 0.0 < lo < 0.10 < hi < 1.0
