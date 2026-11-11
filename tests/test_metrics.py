"""Unit tests for imbalanced classification metrics and bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pytest

from secom_causal.metrics import (
    bootstrap_classification_metrics,
    compute_pr_auc,
    compute_roc_auc,
    recall_at_fixed_precision,
)


def test_perfect_ranking_metrics() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    assert compute_pr_auc(y_true, y_score) == pytest.approx(1.0)
    assert compute_roc_auc(y_true, y_score) == pytest.approx(1.0)

    rec, prec, thresh = recall_at_fixed_precision(y_true, y_score, 0.90)
    assert rec == pytest.approx(1.0)
    assert prec == pytest.approx(1.0)
    assert thresh is not None


def test_recall_at_fixed_precision_unachievable_returns_zero() -> None:
    y_true = np.array([1, 0, 0, 0, 0, 0])
    # Top score belongs to a negative, so precision never reaches 0.80.
    y_score = np.array([0.5, 0.9, 0.4, 0.3, 0.2, 0.1])
    rec, prec, thresh = recall_at_fixed_precision(y_true, y_score, 0.80)
    assert rec == 0.0
    assert prec == 0.0
    assert thresh is None


def test_bootstrap_classification_metrics_structure_and_bounds() -> None:
    rng = np.random.default_rng(7)
    y_true = np.array([0] * 90 + [1] * 10)
    y_score = np.concatenate([rng.uniform(0.0, 0.6, 90), rng.uniform(0.4, 1.0, 10)])

    res = bootstrap_classification_metrics(
        y_true, y_score, precision_targets=[0.20, 0.50], n_bootstrap=200, seed=42
    )
    assert res["n_samples"] == 100
    assert res["n_positives"] == 10
    pr = res["pr_auc"]
    assert 0.0 <= pr["ci_low"] <= pr["estimate"] <= pr["ci_high"] <= 1.0
    roc = res["roc_auc"]
    assert 0.0 <= roc["ci_low"] <= roc["estimate"] <= roc["ci_high"] <= 1.0
    r20 = res["recall_at_precision"]["0.20"]
    assert 0.0 <= r20["ci_low"] <= r20["ci_high"] <= 1.0
