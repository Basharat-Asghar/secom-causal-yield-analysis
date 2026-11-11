"""Leakage-free feature screening via stability selection, mutual information, and collinearity pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ScreeningResult:
    """Output of a single feature-screening configuration."""

    config_name: str
    selected_features: list[str]
    ranking_df: pd.DataFrame
    pruned_collinear_map: dict[str, dict[str, Any]]
    max_abs_corr_selected: float
    condition_number_selected: float


def compute_feature_screening_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_bootstrap: int = 50,
    subsample_fraction: float = 0.75,
    l1_c: float = 0.25,
    tree_top_k: int = 40,
    mi_n_neighbors: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Score all training features via bootstrap stability selection and mutual information.

    Strictly fits on ``X_train`` and ``y_train`` to prevent test leakage.

    Args:
        X_train: Imputed training DataFrame.
        y_train: Binary training labels.
        n_bootstrap: Number of stratified subsampling draws for stability selection.
        subsample_fraction: Fraction of samples drawn per stability replicate.
        l1_c: Inverse L1 regularization strength for logistic stability selection.
        tree_top_k: Top features per XGBoost draw counted as selected.
        mi_n_neighbors: Neighbor count for ``mutual_info_classif``.
        seed: Random seed.

    Returns:
        DataFrame with stability frequencies, MI, and normalized scores.
    """
    features = list(X_train.columns)
    n_features = len(features)
    y_arr = y_train.to_numpy(dtype=int)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        raise ValueError("Need at least 2 positive and 2 negative training rows.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_raw = X_train.to_numpy(dtype=float)

    n_pos_sub = max(2, int(round(len(pos_idx) * subsample_fraction)))
    n_neg_sub = max(2, int(round(len(neg_idx) * subsample_fraction)))
    scale_pos_weight = float(len(neg_idx) / len(pos_idx))

    l1_counts = np.zeros(n_features, dtype=float)
    tree_counts = np.zeros(n_features, dtype=float)
    tree_imp_sum = np.zeros(n_features, dtype=float)

    rng = np.random.default_rng(seed)

    for b in range(n_bootstrap):
        p_draw = rng.choice(pos_idx, size=n_pos_sub, replace=False)
        n_draw = rng.choice(neg_idx, size=n_neg_sub, replace=False)
        sub_idx = np.concatenate([p_draw, n_draw])
        rng.shuffle(sub_idx)

        y_sub = y_arr[sub_idx]

        # 1. Linear L1 stability selection.
        clf_l1 = LogisticRegression(
            C=l1_c,
            l1_ratio=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1500,
            random_state=seed + b,
        )
        clf_l1.fit(X_scaled[sub_idx], y_sub)
        selected_l1 = np.abs(clf_l1.coef_[0]) > 1e-6
        l1_counts += selected_l1.astype(float)

        # 2. Nonlinear XGBoost stability selection.
        clf_tree = XGBClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_alpha=1.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=seed + b,
            n_jobs=1,
        )
        clf_tree.fit(X_raw[sub_idx], y_sub)
        imps = np.asarray(clf_tree.feature_importances_, dtype=float)
        tree_imp_sum += imps
        if np.any(imps > 0):
            top_indices = np.argsort(imps)[::-1][: min(tree_top_k, n_features)]
            top_indices = [idx for idx in top_indices if imps[idx] > 0]
            tree_counts[top_indices] += 1.0

    # 3. Mutual information (averaged over 3 seeds to reduce k-NN jitter).
    mi_accum = np.zeros(n_features, dtype=float)
    for k in range(3):
        mi_accum += mutual_info_classif(
            X_scaled,
            y_arr,
            n_neighbors=mi_n_neighbors,
            random_state=seed + k,
        )
    mi_scores = mi_accum / 3.0

    stability_l1 = l1_counts / float(n_bootstrap)
    stability_tree = tree_counts / float(n_bootstrap)
    mean_tree_imp = tree_imp_sum / float(n_bootstrap)

    mi_max = float(np.max(mi_scores)) if np.max(mi_scores) > 0 else 1.0
    mi_norm = mi_scores / mi_max

    return pd.DataFrame({
        "feature": features,
        "stability_l1": stability_l1,
        "stability_tree": stability_tree,
        "mean_tree_importance": mean_tree_imp,
        "mutual_info": mi_scores,
        "mutual_info_norm": mi_norm,
    })


def select_with_collinearity_pruning(
    X_train: pd.DataFrame,
    scores_df: pd.DataFrame,
    config_name: str,
    target_k: int = 45,
    corr_threshold: float = 0.85,
    w_stability_l1: float = 0.40,
    w_stability_tree: float = 0.35,
    w_mi: float = 0.25,
) -> ScreeningResult:
    """Rank features by weighted stability+MI score and greedily prune collinear duplicates.

    Args:
        X_train: Imputed training DataFrame.
        scores_df: Output of ``compute_feature_screening_scores``.
        config_name: Label for this screening configuration.
        target_k: Exact number of features to retain (if enough non-collinear exist).
        corr_threshold: Maximum allowed absolute Pearson correlation between selected features.
        w_stability_l1: Weight on L1 logistic bootstrap selection frequency.
        w_stability_tree: Weight on tree bootstrap selection frequency.
        w_mi: Weight on normalized mutual information.

    Returns:
        ``ScreeningResult`` containing selected features, ranking table, and pruning audit map.
    """
    df = scores_df.copy()
    df["combined_score"] = (
        w_stability_l1 * df["stability_l1"]
        + w_stability_tree * df["stability_tree"]
        + w_mi * df["mutual_info_norm"]
    )
    df = df.sort_values(
        ["combined_score", "stability_tree", "mutual_info"], ascending=False
    ).reset_index(drop=True)
    df["pre_prune_rank"] = np.arange(1, len(df) + 1)

    corr_matrix = X_train[df["feature"].tolist()].corr(method="pearson").abs().fillna(0.0)

    selected: list[str] = []
    pruned_map: dict[str, dict[str, Any]] = {}

    for feat in df["feature"]:
        if len(selected) >= target_k:
            break
        if not selected:
            selected.append(feat)
            continue
        corrs = corr_matrix.loc[feat, selected]
        max_corr = float(corrs.max())
        if max_corr >= corr_threshold:
            rep = str(corrs.idxmax())
            pruned_map[feat] = {
                "representative": rep,
                "abs_corr": max_corr,
            }
        else:
            selected.append(feat)

    selected_set = set(selected)
    df["selected"] = df["feature"].isin(selected_set)
    df["pruned_as_collinear_to"] = df["feature"].map(
        lambda f: pruned_map[f]["representative"] if f in pruned_map else ""
    )

    if len(selected) >= 2:
        sub_corr = corr_matrix.loc[selected, selected].to_numpy(copy=True)
        np.fill_diagonal(sub_corr, 0.0)
        max_abs_corr = float(np.max(sub_corr))
        X_sel_std = StandardScaler().fit_transform(X_train[selected])
        cond_num = float(np.linalg.cond(X_sel_std))
    else:
        max_abs_corr = 0.0
        cond_num = 1.0

    return ScreeningResult(
        config_name=config_name,
        selected_features=selected,
        ranking_df=df,
        pruned_collinear_map=pruned_map,
        max_abs_corr_selected=max_abs_corr,
        condition_number_selected=cond_num,
    )
