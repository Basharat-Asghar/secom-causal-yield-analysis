"""Unit tests for Phase 2 modeling, time-series CV, and SHAP ranking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from secom_causal.modeling import (
    build_candidate_models,
    compute_shap_values,
    evaluate_time_series_cv,
    rank_features_by_shap,
)


def _make_imbalanced_frame(n: int = 120, p: int = 8, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X_arr = rng.normal(0.0, 1.0, size=(n, p))
    logits = 2.0 * X_arr[:, 0] - 1.5 * X_arr[:, 1] - 2.2
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_arr = (rng.random(n) < probs).astype(int)
    # Guarantee at least a few positives in both halves for TimeSeriesSplit.
    y_arr[5] = 1
    y_arr[-5] = 1
    X = pd.DataFrame(X_arr, columns=[f"sensor_{i}" for i in range(p)])
    y = pd.Series(y_arr, name="fail")
    return X, y


def test_build_candidate_models_has_multiple_families() -> None:
    models = build_candidate_models(scale_pos_weight=10.0, seed=42)
    assert len(models) >= 5
    assert {"lightgbm", "xgboost", "hist_gbm", "random_forest", "logistic_l1", "logistic_l2"}.issubset(
        set(models.keys())
    )


def test_evaluate_time_series_cv_and_shap_shapes() -> None:
    X, y = _make_imbalanced_frame(120, 8, seed=42)
    models = build_candidate_models(scale_pos_weight=5.0, seed=42)

    for name in ("lightgbm", "logistic_l2"):
        est = models[name]
        cv_res = evaluate_time_series_cv(est, X, y, n_splits=3)
        assert cv_res["n_valid_folds"] >= 1
        assert 0.0 <= cv_res["cv_pr_auc_mean"] <= 1.0

        est.fit(X.iloc[:90], y.iloc[:90])
        sv = compute_shap_values(est, X.iloc[:90], X.iloc[90:], seed=42)
        assert sv.shape == (30, 8)

        ranked = rank_features_by_shap(sv, X.iloc[90:])
        assert list(ranked["shap_rank"]) == list(range(1, 9))
        # sensor_0 or sensor_1 should appear among the top features.
        assert len({"sensor_0", "sensor_1"} & set(ranked.head(3)["feature"])) >= 1
