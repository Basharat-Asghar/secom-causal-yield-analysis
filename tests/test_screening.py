"""Unit tests for Phase 3 feature screening and collinearity pruning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from secom_causal.screening import (
    compute_feature_screening_scores,
    select_with_collinearity_pruning,
)


def _make_collinear_signal_data(n: int = 240, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0.0, 1.0, size=n)
    z2 = rng.normal(0.0, 1.0, size=n)
    data = {
        "sig_main": z1,
        "sig_clone": z1 + rng.normal(0.0, 0.05, size=n),
        "sig_second": z2,
    }
    for i in range(15):
        data[f"noise_{i}"] = rng.normal(0.0, 1.0, size=n)
    X = pd.DataFrame(data)
    logits = 2.2 * z1 - 1.6 * z2 - 2.0
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series((rng.random(n) < probs).astype(int), name="fail")
    return X, y


def test_screening_ranks_signal_and_prunes_collinear_duplicate() -> None:
    X, y = _make_collinear_signal_data()
    scores = compute_feature_screening_scores(
        X_train=X,
        y_train=y,
        n_bootstrap=15,
        subsample_fraction=0.75,
        tree_top_k=5,
        seed=42,
    )
    res_strict = select_with_collinearity_pruning(
        X_train=X,
        scores_df=scores,
        config_name="strict",
        target_k=6,
        corr_threshold=0.85,
    )
    assert len(res_strict.selected_features) == 6
    assert res_strict.max_abs_corr_selected < 0.85
    both_in = ("sig_main" in res_strict.selected_features) and ("sig_clone" in res_strict.selected_features)
    assert not both_in
    assert ("sig_main" in res_strict.selected_features) or ("sig_clone" in res_strict.selected_features)
    assert "sig_second" in res_strict.selected_features
