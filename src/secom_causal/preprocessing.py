"""Time-ordered splitting, leakage-free preprocessing, and imputation comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer


ImputationMethod = Literal["median_indicator", "iterative"]


@dataclass(frozen=True)
class SplitData:
    """Container for time-ordered train and test splits."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ts_train: pd.Series
    ts_test: pd.Series
    split_index: int
    split_timestamp: pd.Timestamp


def time_ordered_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    train_fraction: float = 0.70,
) -> SplitData:
    """Split dataset chronologically into contiguous train and test blocks.

    Never shuffles rows. Requires ``timestamps`` to be monotonically
    non-decreasing so test observations strictly follow (or match boundary of)
    training observations.

    Args:
        X: Feature DataFrame sorted chronologically.
        y: Binary target Series aligned with ``X``.
        timestamps: Chronological timestamps aligned with ``X``.
        train_fraction: Fraction of rows assigned to the training block.

    Returns:
        ``SplitData`` instance containing train/test slices and metadata.
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be monotonically non-decreasing before time-ordered split.")
    if not (len(X) == len(y) == len(timestamps)):
        raise ValueError("X, y, and timestamps must have identical lengths.")

    n_total = len(X)
    split_idx = int(np.floor(n_total * train_fraction))
    if split_idx <= 0 or split_idx >= n_total:
        raise ValueError(f"Invalid split_idx={split_idx} for n_total={n_total}.")

    return SplitData(
        X_train=X.iloc[:split_idx].copy(),
        X_test=X.iloc[split_idx:].copy(),
        y_train=y.iloc[:split_idx].copy(),
        y_test=y.iloc[split_idx:].copy(),
        ts_train=timestamps.iloc[:split_idx].copy(),
        ts_test=timestamps.iloc[split_idx:].copy(),
        split_index=split_idx,
        split_timestamp=pd.Timestamp(timestamps.iloc[split_idx]),
    )


class SecomPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-free SECOM feature cleaner and imputer.

    All constant checks, missingness rates, flagged columns, indicator columns,
    and imputer parameters are learned strictly from ``X_train`` during ``fit``.
    """

    def __init__(
        self,
        drop_constant: bool = True,
        high_missing_flag_threshold: float = 0.30,
        high_missing_drop_threshold: float = 0.55,
        imputation_method: ImputationMethod = "median_indicator",
        add_missing_indicators: bool = True,
        indicator_min_missing_rate: float = 0.01,
        iterative_max_iter: int = 10,
        iterative_n_nearest_features: int | None = 20,
        iterative_tol: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.drop_constant = drop_constant
        self.high_missing_flag_threshold = high_missing_flag_threshold
        self.high_missing_drop_threshold = high_missing_drop_threshold
        self.imputation_method = imputation_method
        self.add_missing_indicators = add_missing_indicators
        self.indicator_min_missing_rate = indicator_min_missing_rate
        self.iterative_max_iter = iterative_max_iter
        self.iterative_n_nearest_features = iterative_n_nearest_features
        self.iterative_tol = iterative_tol
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SecomPreprocessor":
        """Learn feature filters, missingness flags, and imputer on training data.

        Args:
            X: Raw training feature DataFrame.
            y: Ignored (present for scikit-learn compatibility).

        Returns:
            Fitted ``SecomPreprocessor`` instance.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.input_features_: list[str] = list(X.columns)
        self.train_missing_rates_: pd.Series = X.isna().mean()

        # Identify constant or all-NaN features on training data only.
        constant_cols: list[str] = []
        if self.drop_constant:
            for col in self.input_features_:
                n_unique = X[col].nunique(dropna=True)
                if n_unique <= 1:
                    constant_cols.append(col)
        self.constant_features_: list[str] = constant_cols

        # Flag & drop high-missingness features (among non-constant, or overall).
        self.flagged_high_missing_: list[str] = [
            col
            for col in self.input_features_
            if col not in self.constant_features_
            and float(self.train_missing_rates_[col]) > self.high_missing_flag_threshold
        ]
        self.dropped_high_missing_: list[str] = [
            col
            for col in self.input_features_
            if col not in self.constant_features_
            and float(self.train_missing_rates_[col]) > self.high_missing_drop_threshold
        ]

        dropped_set = set(self.constant_features_) | set(self.dropped_high_missing_)
        self.kept_sensor_features_: list[str] = [
            col for col in self.input_features_ if col not in dropped_set
        ]

        if not self.kept_sensor_features_:
            raise ValueError("No features survived constant and high-missingness filtering.")

        # Determine missingness indicator columns on training data.
        use_indicators = (
            self.imputation_method == "median_indicator" and self.add_missing_indicators
        )
        if use_indicators:
            self.indicator_features_: list[str] = [
                col
                for col in self.kept_sensor_features_
                if self.indicator_min_missing_rate
                <= float(self.train_missing_rates_[col])
                < 1.0
            ]
        else:
            self.indicator_features_ = []

        X_kept = X[self.kept_sensor_features_]
        if self.imputation_method == "median_indicator":
            self.imputer_: SimpleImputer | IterativeImputer = SimpleImputer(strategy="median")
        elif self.imputation_method == "iterative":
            self.imputer_ = IterativeImputer(
                max_iter=self.iterative_max_iter,
                n_nearest_features=self.iterative_n_nearest_features,
                tol=self.iterative_tol,
                initial_strategy="median",
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported imputation_method: {self.imputation_method}")

        self.imputer_.fit(X_kept)
        self.output_features_: list[str] = list(self.kept_sensor_features_) + [
            f"miss__{col}" for col in self.indicator_features_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned filters and imputation to a DataFrame.

        Args:
            X: Feature DataFrame with columns matching ``fit`` input.

        Returns:
            Imputed DataFrame with kept sensor features (and indicators if enabled).
        """
        if not hasattr(self, "kept_sensor_features_"):
            raise RuntimeError("SecomPreprocessor must be fitted before calling transform.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.input_features_)

        X_kept = X[self.kept_sensor_features_]
        imputed_array = self.imputer_.transform(X_kept)
        df_out = pd.DataFrame(
            imputed_array,
            columns=self.kept_sensor_features_,
            index=X.index,
        )

        if self.indicator_features_:
            miss_df = X[self.indicator_features_].isna().astype(float)
            miss_df.columns = [f"miss__{col}" for col in self.indicator_features_]
            df_out = pd.concat([df_out, miss_df], axis=1)

        return df_out

    def summary(self) -> dict[str, Any]:
        """Return structured summary statistics of fitted preprocessing choices."""
        if not hasattr(self, "kept_sensor_features_"):
            raise RuntimeError("SecomPreprocessor must be fitted before calling summary.")
        return {
            "imputation_method": self.imputation_method,
            "n_input_features": len(self.input_features_),
            "n_constant_dropped": len(self.constant_features_),
            "n_high_missing_flagged": len(self.flagged_high_missing_),
            "n_high_missing_dropped": len(self.dropped_high_missing_),
            "n_kept_sensor_features": len(self.kept_sensor_features_),
            "n_missing_indicators": len(self.indicator_features_),
            "n_output_features": len(self.output_features_),
            "flagged_not_dropped": [
                c for c in self.flagged_high_missing_ if c not in set(self.dropped_high_missing_)
            ],
        }


def summarize_missingness_profile(X_train: pd.DataFrame) -> dict[str, Any]:
    """Compute missingness distribution across raw training features.

    Args:
        X_train: Raw training DataFrame.

    Returns:
        Dictionary of feature counts at standard missingness thresholds.
    """
    miss = X_train.isna().mean()
    n_const = int(sum(X_train[c].nunique(dropna=True) <= 1 for c in X_train.columns))
    return {
        "n_features_total": int(X_train.shape[1]),
        "n_constant_or_all_nan": n_const,
        "overall_cell_missing_rate": float(X_train.isna().mean().mean()),
        "n_zero_missing": int((miss == 0.0).sum()),
        "n_gt_1pct_missing": int((miss > 0.01).sum()),
        "n_gt_10pct_missing": int((miss > 0.10).sum()),
        "n_gt_30pct_missing": int((miss > 0.30).sum()),
        "n_gt_50pct_missing": int((miss > 0.50).sum()),
        "n_gt_70pct_missing": int((miss > 0.70).sum()),
        "n_gt_90pct_missing": int((miss > 0.90).sum()),
    }


def compare_imputation_methods(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    drop_constant: bool = True,
    high_missing_flag_threshold: float = 0.30,
    high_missing_drop_threshold: float = 0.55,
    indicator_min_missing_rate: float = 0.01,
    iterative_max_iter: int = 10,
    iterative_n_nearest_features: int | None = 20,
    iterative_tol: float = 1e-3,
    mask_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare median + missingness indicators vs. iterative imputation.

    Masks a random ``mask_fraction`` of observed entries in columns that have
    natural missingness, fits each imputer strictly on the masked training data,
    and evaluates reconstruction error (NRMSE, MAE) on both masked train and
    masked test entries. Also evaluates missingness-indicator association with
    ``y_train`` via Fisher's exact test.

    Args:
        X_train_raw: Raw training features.
        X_test_raw: Raw test features.
        y_train: Binary training labels.
        drop_constant: Whether to drop constant features.
        high_missing_flag_threshold: Threshold for flagging high-missingness features.
        high_missing_drop_threshold: Threshold for dropping high-missingness features.
        indicator_min_missing_rate: Minimum missing rate for adding indicators.
        iterative_max_iter: Max iterations for ``IterativeImputer``.
        iterative_n_nearest_features: Nearest features for ``IterativeImputer``.
        iterative_tol: Stopping tolerance for ``IterativeImputer``.
        mask_fraction: Fraction of observed cells to artificially mask for benchmark.
        seed: Random seed.

    Returns:
        Dictionary with reconstruction metrics, runtime, and indicator tests.
    """
    base_prep = SecomPreprocessor(
        drop_constant=drop_constant,
        high_missing_flag_threshold=high_missing_flag_threshold,
        high_missing_drop_threshold=high_missing_drop_threshold,
        imputation_method="median_indicator",
        indicator_min_missing_rate=indicator_min_missing_rate,
        random_state=seed,
    ).fit(X_train_raw)

    kept_cols = base_prep.kept_sensor_features_
    X_tr = X_train_raw[kept_cols].copy()
    X_te = X_test_raw[kept_cols].copy()

    # Standard deviations on observed training data for NRMSE normalization.
    col_stds = X_tr.std(ddof=0).replace(0.0, np.nan)
    eval_cols = [c for c in kept_cols if pd.notna(col_stds[c]) and col_stds[c] > 0]

    rng = np.random.default_rng(seed)

    def _mask_observed(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        masked = df.copy()
        arr = masked[eval_cols].to_numpy(copy=True)
        obs_mask = ~np.isnan(arr)
        rand_draw = rng.random(arr.shape)
        hide_mask = obs_mask & (rand_draw < mask_fraction)
        true_vals = arr[hide_mask].copy()
        arr[hide_mask] = np.nan
        masked[eval_cols] = arr
        return masked, hide_mask, true_vals

    X_tr_masked, tr_hide, tr_true = _mask_observed(X_tr)
    X_te_masked, te_hide, te_true = _mask_observed(X_te)

    col_std_matrix_tr = np.broadcast_to(col_stds[eval_cols].to_numpy(), tr_hide.shape)
    col_std_matrix_te = np.broadcast_to(col_stds[eval_cols].to_numpy(), te_hide.shape)
    tr_stds_masked = col_std_matrix_tr[tr_hide]
    te_stds_masked = col_std_matrix_te[te_hide]

    results: dict[str, Any] = {
        "n_eval_features": len(eval_cols),
        "n_masked_train_cells": int(tr_hide.sum()),
        "n_masked_test_cells": int(te_hide.sum()),
        "methods": {},
    }

    for method in ("median_indicator", "iterative"):
        t0 = time.perf_counter()
        if method == "median_indicator":
            imp: SimpleImputer | IterativeImputer = SimpleImputer(strategy="median")
        else:
            imp = IterativeImputer(
                max_iter=iterative_max_iter,
                n_nearest_features=iterative_n_nearest_features,
                tol=iterative_tol,
                initial_strategy="median",
                random_state=seed,
            )
        imp.fit(X_tr_masked)
        tr_imp = pd.DataFrame(imp.transform(X_tr_masked), columns=kept_cols, index=X_tr.index)
        te_imp = pd.DataFrame(imp.transform(X_te_masked), columns=kept_cols, index=X_te.index)
        elapsed = time.perf_counter() - t0

        tr_pred = tr_imp[eval_cols].to_numpy()[tr_hide]
        te_pred = te_imp[eval_cols].to_numpy()[te_hide]

        tr_err = tr_pred - tr_true
        te_err = te_pred - te_true

        results["methods"][method] = {
            "train_nrmse": float(np.sqrt(np.mean((tr_err / tr_stds_masked) ** 2))),
            "train_nmae": float(np.mean(np.abs(tr_err / tr_stds_masked))),
            "test_nrmse": float(np.sqrt(np.mean((te_err / te_stds_masked) ** 2))),
            "test_nmae": float(np.mean(np.abs(te_err / te_stds_masked))),
            "fit_transform_seconds": float(elapsed),
        }

    # Test whether missingness indicators carry univariate signal with y_train.
    sig_indicators: list[dict[str, Any]] = []
    for col in base_prep.indicator_features_:
        is_miss = X_train_raw[col].isna().to_numpy()
        y_arr = y_train.to_numpy()
        table = np.array([
            [int((~is_miss & (y_arr == 0)).sum()), int((~is_miss & (y_arr == 1)).sum())],
            [int((is_miss & (y_arr == 0)).sum()), int((is_miss & (y_arr == 1)).sum())],
        ])
        odds_ratio, p_val = fisher_exact(table)
        if p_val < 0.05:
            sig_indicators.append({
                "feature": col,
                "train_missing_rate": float(is_miss.mean()),
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_val),
            })
    sig_indicators.sort(key=lambda d: d["p_value"])
    results["n_indicator_features"] = len(base_prep.indicator_features_)
    results["n_indicators_fisher_p_lt_0_05"] = len(sig_indicators)
    results["top_associated_indicators"] = sig_indicators[:5]

    return results
