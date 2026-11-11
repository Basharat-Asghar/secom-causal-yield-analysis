"""Candidate predictive models, time-series CV, evaluation, and SHAP baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from secom_causal.metrics import bootstrap_classification_metrics


def build_candidate_models(scale_pos_weight: float, seed: int = 42) -> dict[str, BaseEstimator]:
    """Construct diverse imbalanced-classification candidate models.

    Args:
        scale_pos_weight: Ratio of negative to positive samples in training data.
        seed: Random seed applied to all stochastic estimators.

    Returns:
        Ordered mapping from model name to unfitted scikit-learn estimator.
    """
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=1.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_alpha=1.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=1,
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=5.0,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
        "logistic_l1": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.1,
                    l1_ratio=1.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]),
        "logistic_l2": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.05,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]),
    }


def evaluate_time_series_cv(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Evaluate a candidate estimator using chronological TimeSeriesSplit CV on train.

    Args:
        estimator: Unfitted scikit-learn estimator.
        X_train: Chronologically ordered training features.
        y_train: Binary training labels.
        n_splits: Number of time-series splits.

    Returns:
        Mean and std of fold PR-AUC and ROC-AUC across valid folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pr_scores: list[float] = []
    roc_scores: list[float] = []

    y_arr = y_train.to_numpy()
    for tr_idx, val_idx in tscv.split(X_train):
        y_tr_fold = y_arr[tr_idx]
        y_val_fold = y_arr[val_idx]
        if len(np.unique(y_tr_fold)) < 2 or len(np.unique(y_val_fold)) < 2:
            continue
        model = clone(estimator)
        model.fit(X_train.iloc[tr_idx], y_tr_fold)
        probs = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        pr_scores.append(float(average_precision_score(y_val_fold, probs)))
        roc_scores.append(float(roc_auc_score(y_val_fold, probs)))

    return {
        "n_valid_folds": len(pr_scores),
        "cv_pr_auc_mean": float(np.mean(pr_scores)) if pr_scores else 0.0,
        "cv_pr_auc_std": float(np.std(pr_scores, ddof=1)) if len(pr_scores) > 1 else 0.0,
        "cv_roc_auc_mean": float(np.mean(roc_scores)) if roc_scores else 0.0,
        "cv_roc_auc_std": float(np.std(roc_scores, ddof=1)) if len(roc_scores) > 1 else 0.0,
    }


def _extract_positive_shap(raw_shap: Any) -> np.ndarray:
    """Normalize SHAP output across binary tree/linear explainers to shape (n, p)."""
    if isinstance(raw_shap, list):
        arr = np.asarray(raw_shap[1] if len(raw_shap) == 2 else raw_shap[0], dtype=float)
    else:
        arr = np.asarray(raw_shap, dtype=float)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            arr = arr[:, :, 1]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D positive-class SHAP matrix, got shape {arr.shape}")
    return arr


def compute_shap_values(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray:
    """Compute positive-class SHAP values on ``X_eval`` for tree or linear pipelines.

    Args:
        model: Fitted estimator or Pipeline(scaler, LogisticRegression).
        X_train: Training feature DataFrame (used as background for linear models).
        X_eval: Evaluation feature DataFrame where SHAP values are computed.
        seed: Random seed.

    Returns:
        2D numpy array of shape ``(len(X_eval), X_eval.shape[1])``.
    """
    if isinstance(model, Pipeline):
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        X_bg = scaler.transform(X_train)
        X_ev = scaler.transform(X_eval)
        explainer = shap.LinearExplainer(clf, X_bg, seed=seed)
        return _extract_positive_shap(explainer.shap_values(X_ev))

    explainer = shap.TreeExplainer(model)
    return _extract_positive_shap(explainer.shap_values(X_eval))


def rank_features_by_shap(
    shap_matrix: np.ndarray,
    X_eval: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize SHAP feature importance rankings and directional signs.

    Args:
        shap_matrix: 2D array of SHAP values matching ``X_eval`` columns.
        X_eval: Feature DataFrame corresponding to ``shap_matrix``.

    Returns:
        DataFrame sorted descending by ``mean_abs_shap``.
    """
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    mean_signed = np.mean(shap_matrix, axis=0)
    std_abs = np.std(np.abs(shap_matrix), axis=0, ddof=1)

    corrs: list[float] = []
    for j, col in enumerate(X_eval.columns):
        vals = X_eval[col].to_numpy()
        sv = shap_matrix[:, j]
        if np.std(vals) == 0.0 or np.std(sv) == 0.0:
            corrs.append(0.0)
        else:
            r, _ = spearmanr(vals, sv)
            corrs.append(float(r) if np.isfinite(r) else 0.0)

    df = pd.DataFrame({
        "feature": list(X_eval.columns),
        "mean_abs_shap": mean_abs,
        "std_abs_shap": std_abs,
        "mean_shap": mean_signed,
        "spearman_value_vs_shap": corrs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["shap_rank"] = np.arange(1, len(df) + 1)
    return df


def save_phase2_figures(
    comparison_df: pd.DataFrame,
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    shap_df: pd.DataFrame,
    champion_name: str,
    base_rate: float,
    figures_dir: str | Path,
) -> list[Path]:
    """Save Phase 2 model comparison and champion SHAP ranking plots."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Model PR-AUC comparison with 95% bootstrap CIs + PR curves.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    df_plot = comparison_df.sort_values("test_pr_auc", ascending=True)
    y_pos = np.arange(len(df_plot))
    est = df_plot["test_pr_auc"].to_numpy()
    err_lo = est - df_plot["test_pr_auc_ci_low"].to_numpy()
    err_hi = df_plot["test_pr_auc_ci_high"].to_numpy() - est
    colors = ["#1f4e78" if lbl == champion_name else "#7f8c8d" for lbl in df_plot["model_id"]]

    axes[0].barh(y_pos, est, xerr=[err_lo, err_hi], capsize=4, color=colors, alpha=0.9)
    axes[0].axvline(base_rate, color="#b85450", linestyle="--", label=f"No-skill ({base_rate:.3f})")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_plot["model_id"])
    axes[0].set_xlabel("Test PR-AUC (95% Bootstrap CI)")
    axes[0].set_title("Model Benchmark on Time-Ordered Test Set")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.3)

    for model_id, (prec, rec) in pr_curves.items():
        lw = 2.5 if model_id == champion_name else 1.2
        alpha = 1.0 if model_id == champion_name else 0.7
        axes[1].plot(rec, prec, linewidth=lw, alpha=alpha, label=model_id)
    axes[1].axhline(base_rate, color="#b85450", linestyle="--", linewidth=1, label="Base rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves (Test Set)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    comp_fig = out_dir / "phase2_model_comparison.png"
    fig.savefig(comp_fig, dpi=150)
    plt.close(fig)
    saved.append(comp_fig)

    # 2. Top-20 SHAP feature importance for champion model.
    top20 = shap_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = [
        "#b85450" if r >= 0 else "#2e6b9e"
        for r in top20["spearman_value_vs_shap"]
    ]
    ax.barh(top20["feature"], top20["mean_abs_shap"], color=bar_colors)
    ax.set_xlabel("Mean |SHAP value| (Test Set)")
    ax.set_title(f"Champion ({champion_name}) Top 20 Features by Mean |SHAP|\n(Red: higher value -> higher fail risk; Blue: inverse)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    shap_fig = out_dir / "phase2_shap_top20.png"
    fig.savefig(shap_fig, dpi=150)
    plt.close(fig)
    saved.append(shap_fig)

    return saved
