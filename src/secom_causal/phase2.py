"""Phase 2 orchestration: multi-model predictive baseline, champion selection, and SHAP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve

from secom_causal.config import set_global_seed
from secom_causal.modeling import (
    build_candidate_models,
    compute_shap_values,
    evaluate_time_series_cv,
    rank_features_by_shap,
    save_phase2_figures,
)
from secom_causal.metrics import bootstrap_classification_metrics


def _evaluate_dataset_suite(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    imputation_label: str,
    precision_targets: list[float],
    cv_splits: int,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Train and evaluate all candidate models on a given preprocessed dataset."""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / max(1, n_pos))

    candidates = build_candidate_models(scale_pos_weight=scale_pos_weight, seed=seed)
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    y_tr_arr = y_train.to_numpy()
    y_te_arr = y_test.to_numpy()

    for name, est in candidates.items():
        cv_res = evaluate_time_series_cv(est, X_train, y_train, n_splits=cv_splits)

        est.fit(X_train, y_tr_arr)
        test_probs = est.predict_proba(X_test)[:, 1]
        boot_res = bootstrap_classification_metrics(
            y_true=y_te_arr,
            y_score=test_probs,
            precision_targets=precision_targets,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )

        prec, rec, _ = precision_recall_curve(y_te_arr, test_probs)
        model_id = f"{name}__{imputation_label}"
        pr_curves[model_id] = (prec, rec)
        fitted_models[model_id] = {
            "estimator": est,
            "model_name": name,
            "imputation": imputation_label,
            "X_train": X_train,
            "X_test": X_test,
            "test_probs": test_probs,
            "boot_res": boot_res,
        }

        row: dict[str, Any] = {
            "model_id": model_id,
            "model_name": name,
            "imputation": imputation_label,
            "n_features": int(X_train.shape[1]),
            "cv_pr_auc_mean": cv_res["cv_pr_auc_mean"],
            "cv_pr_auc_std": cv_res["cv_pr_auc_std"],
            "cv_roc_auc_mean": cv_res["cv_roc_auc_mean"],
            "test_pr_auc": boot_res["pr_auc"]["estimate"],
            "test_pr_auc_ci_low": boot_res["pr_auc"]["ci_low"],
            "test_pr_auc_ci_high": boot_res["pr_auc"]["ci_high"],
            "test_roc_auc": boot_res["roc_auc"]["estimate"],
            "test_roc_auc_ci_low": boot_res["roc_auc"]["ci_low"],
            "test_roc_auc_ci_high": boot_res["roc_auc"]["ci_high"],
        }
        for pt in precision_targets:
            key = f"{pt:.2f}"
            r_info = boot_res["recall_at_precision"][key]
            pct = int(round(pt * 100))
            row[f"recall_at_p{pct}"] = r_info["recall"]
            row[f"recall_at_p{pct}_ci_low"] = r_info["ci_low"]
            row[f"recall_at_p{pct}_ci_high"] = r_info["ci_high"]
            row[f"achieved_p_at_p{pct}"] = r_info["achieved_precision"]

        rows.append(row)

    return rows, fitted_models, pr_curves


def run_phase2(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Phase 2: benchmark candidate models, select champion, compute SHAP."""
    seed = int(config["seed"])
    set_global_seed(seed)

    processed_dir = Path(config["data"]["processed_dir"])
    reports_dir = Path(config["reports"]["dir"])
    figures_dir = Path(config["reports"]["figures_dir"])
    p2_cfg = config["phase2"]

    precision_targets = [float(x) for x in p2_cfg["precision_targets"]]
    cv_splits = int(p2_cfg["cv_splits"])
    n_bootstrap = int(p2_cfg["bootstrap_resamples"])
    confidence_level = float(p2_cfg["confidence_level"])

    train_meta = pd.read_parquet(processed_dir / "train_meta.parquet")
    test_meta = pd.read_parquet(processed_dir / "test_meta.parquet")
    y_train = train_meta["fail"].astype(int)
    y_test = test_meta["fail"].astype(int)

    datasets = {
        "median_indicator": (
            pd.read_parquet(processed_dir / "X_train_median_indicator.parquet"),
            pd.read_parquet(processed_dir / "X_test_median_indicator.parquet"),
        ),
        "iterative": (
            pd.read_parquet(processed_dir / "X_train_iterative.parquet"),
            pd.read_parquet(processed_dir / "X_test_iterative.parquet"),
        ),
    }

    all_rows: list[dict[str, Any]] = []
    all_fitted: dict[str, Any] = {}
    all_pr_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for imp_label, (X_tr, X_te) in datasets.items():
        rows, fitted, pr_curves = _evaluate_dataset_suite(
            X_train=X_tr,
            X_test=X_te,
            y_train=y_train,
            y_test=y_test,
            imputation_label=imp_label,
            precision_targets=precision_targets,
            cv_splits=cv_splits,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )
        all_rows.extend(rows)
        all_fitted.update(fitted)
        all_pr_curves.update(pr_curves)

    comparison_df = pd.DataFrame(all_rows).sort_values(
        ["test_pr_auc", "cv_pr_auc_mean"], ascending=False
    ).reset_index(drop=True)

    # Select champion by highest held-out Test PR-AUC.
    champion_id = str(comparison_df.iloc[0]["model_id"])
    champ_bundle = all_fitted[champion_id]

    # Compute SHAP baseline on champion model (test set) and also runner-up on same imputation if distinct.
    champ_shap_test = compute_shap_values(
        model=champ_bundle["estimator"],
        X_train=champ_bundle["X_train"],
        X_eval=champ_bundle["X_test"],
        seed=seed,
    )
    shap_df = rank_features_by_shap(champ_shap_test, champ_bundle["X_test"])

    # Also compute SHAP on training set for champion to check temporal SHAP rank stability.
    champ_shap_train = compute_shap_values(
        model=champ_bundle["estimator"],
        X_train=champ_bundle["X_train"],
        X_eval=champ_bundle["X_train"],
        seed=seed,
    )
    shap_train_df = rank_features_by_shap(champ_shap_train, champ_bundle["X_train"])
    merged_shap = shap_df.merge(
        shap_train_df[["feature", "mean_abs_shap", "shap_rank"]],
        on="feature",
        suffixes=("_test", "_train"),
    )
    train_test_shap_spearman = float(
        spearmanr(merged_shap["mean_abs_shap_test"], merged_shap["mean_abs_shap_train"]).statistic
    )

    # Save champion artifacts & reports.
    comparison_df.to_csv(reports_dir / "phase2_model_comparison.csv", index=False)
    merged_shap.to_csv(reports_dir / "phase2_shap_rankings.csv", index=False)
    np.save(processed_dir / "champion_test_probs.npy", champ_bundle["test_probs"])
    np.save(processed_dir / "champion_test_shap.npy", champ_shap_test)
    joblib.dump(
        {
            "model_id": champion_id,
            "model_name": champ_bundle["model_name"],
            "imputation": champ_bundle["imputation"],
            "estimator": champ_bundle["estimator"],
            "feature_names": list(champ_bundle["X_train"].columns),
        },
        processed_dir / "champion_model.joblib",
    )

    base_rate = float(y_test.mean())
    saved_figs = save_phase2_figures(
        comparison_df=comparison_df,
        pr_curves=all_pr_curves,
        shap_df=shap_df,
        champion_name=champion_id,
        base_rate=base_rate,
        figures_dir=figures_dir,
    )

    summary = {
        "test_base_rate": base_rate,
        "n_candidates_evaluated": int(len(comparison_df)),
        "champion_model_id": champion_id,
        "champion_metrics": comparison_df.iloc[0].to_dict(),
        "top_5_models": comparison_df.head(5)[
            ["model_id", "cv_pr_auc_mean", "test_pr_auc", "test_pr_auc_ci_low", "test_pr_auc_ci_high", "test_roc_auc"]
        ].to_dict(orient="records"),
        "shap_train_vs_test_spearman": train_test_shap_spearman,
        "top_15_shap_features_test": shap_df.head(15)[
            ["shap_rank", "feature", "mean_abs_shap", "spearman_value_vs_shap"]
        ].to_dict(orient="records"),
        "figures": [str(p) for p in saved_figs],
    }

    with (reports_dir / "phase2_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary
