"""Phase 3 orchestration: feature screening and downstream sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from secom_causal.config import set_global_seed
from secom_causal.metrics import bootstrap_classification_metrics
from secom_causal.modeling import (
    build_candidate_models,
    compute_shap_values,
    evaluate_time_series_cv,
    rank_features_by_shap,
)
from secom_causal.screening import (
    ScreeningResult,
    compute_feature_screening_scores,
    select_with_collinearity_pruning,
)


def _save_phase3_figures(
    sensitivity_df: pd.DataFrame,
    results_map: dict[str, ScreeningResult],
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    default_res = next(iter(results_map.values()))
    r_df = default_res.ranking_df

    sel_mask = r_df["selected"]
    axes[0].scatter(
        r_df.loc[~sel_mask, "stability_l1"],
        r_df.loc[~sel_mask, "stability_tree"],
        alpha=0.35,
        s=18,
        color="#95a5a6",
        label="Not selected",
    )
    axes[0].scatter(
        r_df.loc[sel_mask, "stability_l1"],
        r_df.loc[sel_mask, "stability_tree"],
        alpha=0.85,
        s=32,
        color="#1f4e78",
        label=f"Selected ({default_res.config_name}, k={len(default_res.selected_features)})",
    )
    axes[0].set_xlabel("Bootstrap L1 Stability Frequency")
    axes[0].set_ylabel("Bootstrap Tree Stability Frequency")
    axes[0].set_title("Stability Selection Landscape (Train Only)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    df_plot = sensitivity_df.sort_values("test_pr_auc", ascending=True)
    y_pos = np.arange(len(df_plot))
    est = df_plot["test_pr_auc"].to_numpy()
    err_lo = est - df_plot["test_pr_auc_ci_low"].to_numpy()
    err_hi = df_plot["test_pr_auc_ci_high"].to_numpy() - est
    labels = [f"{r.screening_config} | {r.model_name} (p={r.n_features})" for r in df_plot.itertuples()]

    axes[1].barh(y_pos, est, xerr=[err_lo, err_hi], capsize=3, color="#2e6b9e", alpha=0.9)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Test PR-AUC (95% Bootstrap CI)")
    axes[1].set_title("Downstream Sensitivity Across Screening Configs")
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig_path = figures_dir / "phase3_screening_sensitivity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    saved.append(fig_path)
    return saved


def run_phase3(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Phase 3 screening and downstream sensitivity checks."""
    seed = int(config["seed"])
    set_global_seed(seed)

    processed_dir = Path(config["data"]["processed_dir"])
    reports_dir = Path(config["reports"]["dir"])
    figures_dir = Path(config["reports"]["figures_dir"])
    p2_cfg = config["phase2"]
    p3_cfg = config["phase3"]

    imp_name = str(p3_cfg["primary_imputation"])
    X_train = pd.read_parquet(processed_dir / f"X_train_{imp_name}.parquet")
    X_test = pd.read_parquet(processed_dir / f"X_test_{imp_name}.parquet")
    train_meta = pd.read_parquet(processed_dir / "train_meta.parquet")
    test_meta = pd.read_parquet(processed_dir / "test_meta.parquet")
    y_train = train_meta["fail"].astype(int)
    y_test = test_meta["fail"].astype(int)

    # 1. Score all features strictly on X_train, y_train.
    scores_df = compute_feature_screening_scores(
        X_train=X_train,
        y_train=y_train,
        n_bootstrap=int(p3_cfg["n_bootstrap"]),
        subsample_fraction=float(p3_cfg["subsample_fraction"]),
        l1_c=float(p3_cfg["l1_c"]),
        tree_top_k=int(p3_cfg["tree_top_k"]),
        mi_n_neighbors=int(p3_cfg["mi_n_neighbors"]),
        seed=seed,
    )

    # 2. Apply each screening configuration + collinearity pruning.
    screening_results: dict[str, ScreeningResult] = {}
    config_summaries: list[dict[str, Any]] = []
    for cfg_name, cfg_params in p3_cfg["configs"].items():
        res = select_with_collinearity_pruning(
            X_train=X_train,
            scores_df=scores_df,
            config_name=cfg_name,
            target_k=int(cfg_params["target_k"]),
            corr_threshold=float(cfg_params["corr_threshold"]),
            w_stability_l1=float(cfg_params["w_stability_l1"]),
            w_stability_tree=float(cfg_params["w_stability_tree"]),
            w_mi=float(cfg_params["w_mi"]),
        )
        screening_results[cfg_name] = res
        res.ranking_df.to_csv(reports_dir / f"phase3_screening_{cfg_name}.csv", index=False)

        X_train[res.selected_features].to_parquet(
            processed_dir / f"X_train_screened_{cfg_name}.parquet"
        )
        X_test[res.selected_features].to_parquet(
            processed_dir / f"X_test_screened_{cfg_name}.parquet"
        )

        config_summaries.append({
            "config_name": cfg_name,
            "target_k": int(cfg_params["target_k"]),
            "n_selected": len(res.selected_features),
            "n_pruned_collinear": len(res.pruned_collinear_map),
            "corr_threshold": float(cfg_params["corr_threshold"]),
            "max_abs_corr_selected": res.max_abs_corr_selected,
            "condition_number_selected": res.condition_number_selected,
        })

    # 3. Pairwise overlap across screening configurations.
    cfg_names = list(screening_results.keys())
    overlap_matrix: list[dict[str, Any]] = []
    for i, c1 in enumerate(cfg_names):
        s1 = set(screening_results[c1].selected_features)
        for c2 in cfg_names[i + 1 :]:
            s2 = set(screening_results[c2].selected_features)
            inter = len(s1 & s2)
            union = len(s1 | s2)
            overlap_matrix.append({
                "config_a": c1,
                "config_b": c2,
                "n_a": len(s1),
                "n_b": len(s2),
                "intersection": inter,
                "jaccard": float(inter / union) if union > 0 else 0.0,
            })

    # 4. Downstream sensitivity check (Phase 2 champion xgboost + random_forest + logistic_l2).
    phase2_shap_df = pd.read_csv(reports_dir / "phase2_shap_rankings.csv")
    phase2_top20 = set(phase2_shap_df.head(20)["feature"].tolist())

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / max(1, n_pos))
    precision_targets = [float(x) for x in p2_cfg["precision_targets"]]
    n_bootstrap = int(p2_cfg["bootstrap_resamples"])
    conf_level = float(p2_cfg["confidence_level"])
    cv_splits = int(p2_cfg["cv_splits"])

    eval_models = ["xgboost", "random_forest", "logistic_l2"]
    sensitivity_rows: list[dict[str, Any]] = []

    for cfg_name, res in screening_results.items():
        X_tr_sub = X_train[res.selected_features]
        X_te_sub = X_test[res.selected_features]
        candidates = build_candidate_models(scale_pos_weight=scale_pos_weight, seed=seed)

        for model_name in eval_models:
            est = candidates[model_name]
            cv_res = evaluate_time_series_cv(est, X_tr_sub, y_train, n_splits=cv_splits)
            est.fit(X_tr_sub, y_train.to_numpy())
            probs = est.predict_proba(X_te_sub)[:, 1]
            boot = bootstrap_classification_metrics(
                y_true=y_test.to_numpy(),
                y_score=probs,
                precision_targets=precision_targets,
                n_bootstrap=n_bootstrap,
                confidence_level=conf_level,
                seed=seed,
            )

            shap_overlap_20 = np.nan
            shap_spearman_vs_p2 = np.nan
            if model_name == "xgboost":
                sv = compute_shap_values(est, X_tr_sub, X_te_sub, seed=seed)
                s_df = rank_features_by_shap(sv, X_te_sub)
                shap_overlap_20 = len(set(s_df.head(20)["feature"]) & phase2_top20)
                joined = s_df.merge(
                    phase2_shap_df[["feature", "mean_abs_shap_test"]],
                    on="feature",
                    how="inner",
                )
                if len(joined) >= 5:
                    shap_spearman_vs_p2 = float(
                        spearmanr(joined["mean_abs_shap"], joined["mean_abs_shap_test"]).statistic
                    )

            sensitivity_rows.append({
                "screening_config": cfg_name,
                "model_name": model_name,
                "n_features": len(res.selected_features),
                "cv_pr_auc_mean": cv_res["cv_pr_auc_mean"],
                "test_pr_auc": boot["pr_auc"]["estimate"],
                "test_pr_auc_ci_low": boot["pr_auc"]["ci_low"],
                "test_pr_auc_ci_high": boot["pr_auc"]["ci_high"],
                "test_roc_auc": boot["roc_auc"]["estimate"],
                "test_roc_auc_ci_low": boot["roc_auc"]["ci_low"],
                "test_roc_auc_ci_high": boot["roc_auc"]["ci_high"],
                "phase2_top20_shap_overlap": shap_overlap_20,
                "phase2_shap_spearman": shap_spearman_vs_p2,
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(reports_dir / "phase3_downstream_sensitivity.csv", index=False)

    saved_figs = _save_phase3_figures(
        sensitivity_df=sensitivity_df,
        results_map=screening_results,
        figures_dir=figures_dir,
    )

    default_cfg = str(p3_cfg["default_screened_config"])
    summary = {
        "primary_imputation": imp_name,
        "n_input_features": int(X_train.shape[1]),
        "default_screened_config": default_cfg,
        "configs": config_summaries,
        "pairwise_overlap": overlap_matrix,
        "phase2_top20_in_default_screened": len(
            set(screening_results[default_cfg].selected_features) & phase2_top20
        ),
        "downstream_sensitivity": sensitivity_rows,
        "default_top_15_features": screening_results[default_cfg]
        .ranking_df[screening_results[default_cfg].ranking_df["selected"]]
        .head(15)[["feature", "combined_score", "stability_l1", "stability_tree", "mutual_info"]]
        .to_dict(orient="records"),
        "figures": [str(p) for p in saved_figs],
    }

    with (reports_dir / "phase3_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary
