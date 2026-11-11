"""Classification metrics and stratified bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the precision-recall curve (Average Precision).

    Args:
        y_true: Binary ground truth array in {0, 1}.
        y_score: Predicted probability or decision score for positive class.

    Returns:
        Average precision score in [0, 1].
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("y_true must contain both positive and negative classes.")
    return float(average_precision_score(y_true_arr, y_score_arr))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC score.

    Args:
        y_true: Binary ground truth array in {0, 1}.
        y_score: Predicted probability or decision score for positive class.

    Returns:
        ROC-AUC score in [0, 1].
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("y_true must contain both positive and negative classes.")
    return float(roc_auc_score(y_true_arr, y_score_arr))


def recall_at_fixed_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float,
) -> tuple[float, float, float | None]:
    """Compute maximum recall achievable while maintaining precision >= target.

    Args:
        y_true: Binary ground truth array in {0, 1}.
        y_score: Predicted positive scores.
        target_precision: Minimum required precision in (0, 1].

    Returns:
        Tuple ``(recall, achieved_precision, threshold)``. If no threshold
        achieves ``target_precision`` with non-zero recall, returns
        ``(0.0, 0.0, None)``.
    """
    if not (0.0 < target_precision <= 1.0):
        raise ValueError(f"target_precision must be in (0, 1], got {target_precision}")

    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_score_arr)

    # Ignore the final sentinel point (precision=1.0, recall=0.0) without threshold.
    p_valid = precisions[:-1]
    r_valid = recalls[:-1]
    qualifying = np.where((p_valid >= target_precision) & (r_valid > 0.0))[0]
    if len(qualifying) == 0:
        return (0.0, 0.0, None)

    best_idx = int(qualifying[np.argmax(r_valid[qualifying])])
    return (
        float(r_valid[best_idx]),
        float(p_valid[best_idx]),
        float(thresholds[best_idx]),
    )


def bootstrap_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    precision_targets: list[float] | tuple[float, ...] = (0.15, 0.20, 0.30),
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute point estimates and stratified bootstrap CIs for imbalanced metrics.

    Stratified resampling preserves exact positive and negative counts on each
    bootstrap draw so rare-event metrics remain well-defined.

    Args:
        y_true: Binary ground truth array in {0, 1}.
        y_score: Predicted scores for positive class.
        precision_targets: Precision thresholds for recall-at-precision evaluation.
        n_bootstrap: Number of bootstrap replicates.
        confidence_level: Two-sided confidence level (e.g. 0.95).
        seed: Random seed.

    Returns:
        Dictionary of point estimates, standard deviations, and percentile CIs.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    pos_idx = np.where(y_true_arr == 1)[0]
    neg_idx = np.where(y_true_arr == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Both positive and negative samples are required for stratified bootstrap.")

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level
    q_low = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)

    pr_point = compute_pr_auc(y_true_arr, y_score_arr)
    roc_point = compute_roc_auc(y_true_arr, y_score_arr)
    recall_points: dict[float, tuple[float, float, float | None]] = {
        pt: recall_at_fixed_precision(y_true_arr, y_score_arr, pt)
        for pt in precision_targets
    }

    pr_boot = np.empty(n_bootstrap, dtype=float)
    roc_boot = np.empty(n_bootstrap, dtype=float)
    rec_boot: dict[float, np.ndarray] = {
        pt: np.empty(n_bootstrap, dtype=float) for pt in precision_targets
    }

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    for b in range(n_bootstrap):
        draw_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        draw_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([draw_pos, draw_neg])
        yb = y_true_arr[idx]
        sb = y_score_arr[idx]

        pr_boot[b] = average_precision_score(yb, sb)
        roc_boot[b] = roc_auc_score(yb, sb)
        for pt in precision_targets:
            rec_boot[pt][b] = recall_at_fixed_precision(yb, sb, pt)[0]

    recall_summary: dict[str, dict[str, Any]] = {}
    for pt in precision_targets:
        rec_val, achieved_p, thresh = recall_points[pt]
        arr = rec_boot[pt]
        recall_summary[f"{pt:.2f}"] = {
            "target_precision": float(pt),
            "recall": rec_val,
            "achieved_precision": achieved_p,
            "threshold": thresh,
            "ci_low": float(np.percentile(arr, q_low)),
            "ci_high": float(np.percentile(arr, q_high)),
            "boot_std": float(np.std(arr, ddof=1)),
        }

    return {
        "n_samples": int(len(y_true_arr)),
        "n_positives": int(n_pos),
        "base_rate": float(n_pos / len(y_true_arr)),
        "pr_auc": {
            "estimate": pr_point,
            "ci_low": float(np.percentile(pr_boot, q_low)),
            "ci_high": float(np.percentile(pr_boot, q_high)),
            "boot_std": float(np.std(pr_boot, ddof=1)),
        },
        "roc_auc": {
            "estimate": roc_point,
            "ci_low": float(np.percentile(roc_boot, q_low)),
            "ci_high": float(np.percentile(roc_boot, q_high)),
            "boot_std": float(np.std(roc_boot, ddof=1)),
        },
        "recall_at_precision": recall_summary,
    }
