"""Phase 1 orchestration: download, preprocessing, imputation comparison, and drift analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from secom_causal.config import set_global_seed
from secom_causal.data_loader import download_secom, load_secom
from secom_causal.drift import analyze_temporal_drift, save_phase1_figures
from secom_causal.preprocessing import (
    SecomPreprocessor,
    compare_imputation_methods,
    summarize_missingness_profile,
    time_ordered_split,
)


def run_phase1(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Phase 1 end-to-end and persist processed datasets and reports.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Structured Phase 1 summary dictionary.
    """
    seed = int(config["seed"])
    set_global_seed(seed)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    drift_cfg = config["drift"]
    reports_cfg = config["reports"]

    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    reports_dir = Path(reports_cfg["dir"])
    figures_dir = Path(reports_cfg["figures_dir"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download & load raw SECOM data.
    download_secom(raw_dir=raw_dir, url=data_cfg["uci_zip_url"])
    X, y, timestamps, load_info = load_secom(
        raw_dir=raw_dir,
        data_file=data_cfg["secom_data_file"],
        labels_file=data_cfg["secom_labels_file"],
        sort_by_time=True,
    )

    # 2. Time-ordered train/test split.
    split = time_ordered_split(
        X=X,
        y=y,
        timestamps=timestamps,
        train_fraction=float(prep_cfg["train_fraction"]),
    )

    # 3. Missingness profile on training data only.
    missing_profile = summarize_missingness_profile(split.X_train)

    # 4. Fit primary preprocessor (median + missingness indicators) and iterative comparison.
    iter_cfg = prep_cfg["iterative_imputation"]
    prep_median = SecomPreprocessor(
        drop_constant=bool(prep_cfg["drop_constant"]),
        high_missing_flag_threshold=float(prep_cfg["high_missing_flag_threshold"]),
        high_missing_drop_threshold=float(prep_cfg["high_missing_drop_threshold"]),
        imputation_method="median_indicator",
        add_missing_indicators=bool(prep_cfg["add_missing_indicators"]),
        indicator_min_missing_rate=float(prep_cfg["indicator_min_missing_rate"]),
        random_state=seed,
    ).fit(split.X_train)

    prep_iter = SecomPreprocessor(
        drop_constant=bool(prep_cfg["drop_constant"]),
        high_missing_flag_threshold=float(prep_cfg["high_missing_flag_threshold"]),
        high_missing_drop_threshold=float(prep_cfg["high_missing_drop_threshold"]),
        imputation_method="iterative",
        add_missing_indicators=False,
        iterative_max_iter=int(iter_cfg["max_iter"]),
        iterative_n_nearest_features=int(iter_cfg["n_nearest_features"]),
        iterative_tol=float(iter_cfg["tol"]),
        random_state=seed,
    ).fit(split.X_train)

    X_train_med = prep_median.transform(split.X_train)
    X_test_med = prep_median.transform(split.X_test)
    X_train_iter = prep_iter.transform(split.X_train)
    X_test_iter = prep_iter.transform(split.X_test)

    # 5. Imputation comparison benchmark (masked observed values, no leakage).
    imp_comparison = compare_imputation_methods(
        X_train_raw=split.X_train,
        X_test_raw=split.X_test,
        y_train=split.y_train,
        drop_constant=bool(prep_cfg["drop_constant"]),
        high_missing_flag_threshold=float(prep_cfg["high_missing_flag_threshold"]),
        high_missing_drop_threshold=float(prep_cfg["high_missing_drop_threshold"]),
        indicator_min_missing_rate=float(prep_cfg["indicator_min_missing_rate"]),
        iterative_max_iter=int(iter_cfg["max_iter"]),
        iterative_n_nearest_features=int(iter_cfg["n_nearest_features"]),
        iterative_tol=float(iter_cfg["tol"]),
        seed=seed,
    )

    # 6. Temporal & covariate drift analysis.
    bins_df, feature_drift_df, drift_summary = analyze_temporal_drift(
        X_train_raw=split.X_train,
        X_test_raw=split.X_test,
        y=y,
        timestamps=timestamps,
        kept_features=prep_median.kept_sensor_features_,
        n_time_bins=int(drift_cfg["n_time_bins"]),
        psi_warn_threshold=float(drift_cfg["psi_warn_threshold"]),
        ks_alpha=float(drift_cfg["ks_alpha"]),
    )

    saved_figs = save_phase1_figures(
        bins_df=bins_df,
        feature_drift_df=feature_drift_df,
        raw_missing_rates=prep_median.train_missing_rates_,
        figures_dir=figures_dir,
    )

    # 7. Persist processed artifacts.
    X_train_med.to_parquet(processed_dir / "X_train_median_indicator.parquet")
    X_test_med.to_parquet(processed_dir / "X_test_median_indicator.parquet")
    X_train_iter.to_parquet(processed_dir / "X_train_iterative.parquet")
    X_test_iter.to_parquet(processed_dir / "X_test_iterative.parquet")

    train_meta = pd.DataFrame({
        "fail": split.y_train.to_numpy(),
        "timestamp": split.ts_train.to_numpy(),
    }, index=split.X_train.index)
    test_meta = pd.DataFrame({
        "fail": split.y_test.to_numpy(),
        "timestamp": split.ts_test.to_numpy(),
    }, index=split.X_test.index)
    train_meta.to_parquet(processed_dir / "train_meta.parquet")
    test_meta.to_parquet(processed_dir / "test_meta.parquet")

    bins_df.to_csv(reports_dir / "phase1_temporal_bins.csv", index=False)
    feature_drift_df.to_csv(reports_dir / "phase1_feature_drift.csv", index=False)

    summary = {
        "load_info": load_info,
        "split_info": {
            "train_fraction": float(prep_cfg["train_fraction"]),
            "split_index": split.split_index,
            "train_start": str(split.ts_train.iloc[0]),
            "train_end": str(split.ts_train.iloc[-1]),
            "test_start": str(split.ts_test.iloc[0]),
            "test_end": str(split.ts_test.iloc[-1]),
            "train_n": int(len(split.X_train)),
            "train_failures": int(split.y_train.sum()),
            "train_fail_rate": float(split.y_train.mean()),
            "test_n": int(len(split.X_test)),
            "test_failures": int(split.y_test.sum()),
            "test_fail_rate": float(split.y_test.mean()),
        },
        "missingness_profile_train": missing_profile,
        "preprocessor_median_indicator": prep_median.summary(),
        "preprocessor_iterative": prep_iter.summary(),
        "imputation_comparison": imp_comparison,
        "drift_summary": drift_summary,
        "figures": [str(p) for p in saved_figs],
    }

    summary_path = reports_dir / "phase1_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary
