"""Temporal process drift and label-rate analysis for UCI SECOM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of positive events.
        n: Total trials.
        z: Normal quantile (1.96 for 95% CI).

    Returns:
        Tuple ``(lower, upper)`` in ``[0, 1]``.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * n)) / n)) / denom
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and target arrays.

    Quantile bin edges are derived strictly from ``expected`` (training data).

    Args:
        expected: 1D reference array (finite values).
        actual: 1D comparison array (finite values).
        n_bins: Number of quantile bins.

    Returns:
        Non-negative PSI scalar. Returns 0.0 if ``expected`` has zero spread.
    """
    exp_clean = expected[np.isfinite(expected)]
    act_clean = actual[np.isfinite(actual)]
    if len(exp_clean) < 2 or len(act_clean) < 2:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(exp_clean, quantiles))
    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_clean, bins=edges)
    act_counts, _ = np.histogram(act_clean, bins=edges)

    eps = 1e-4
    exp_pct = np.clip(exp_counts / len(exp_clean), eps, None)
    act_pct = np.clip(act_counts / len(act_clean), eps, None)
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR-adjusted q-values.

    Args:
        p_values: 1D array of raw p-values.

    Returns:
        1D array of adjusted q-values in ``[0, 1]``.
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    q = ranked * n / np.arange(1, n + 1)
    q_monotone = np.minimum.accumulate(q[::-1])[::-1]
    q_monotone = np.clip(q_monotone, 0.0, 1.0)
    out = np.empty_like(q_monotone)
    out[order] = q_monotone
    return out


def analyze_temporal_drift(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    kept_features: list[str],
    n_time_bins: int = 5,
    psi_warn_threshold: float = 0.25,
    ks_alpha: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Analyze yield-rate, missingness, and feature-distribution drift over time.

    Args:
        X_train_raw: Raw training features (kept or all columns).
        X_test_raw: Raw test features.
        y: Full chronologically sorted target Series.
        timestamps: Full chronologically sorted timestamp Series.
        kept_features: List of surviving sensor feature names to test for drift.
        n_time_bins: Number of chronological bins for temporal profile.
        psi_warn_threshold: PSI threshold flagging notable covariate shift.
        ks_alpha: FDR q-value threshold for KS distribution shift flag.

    Returns:
        Tuple ``(bins_df, feature_drift_df, summary_dict)``.
    """
    n_total = len(y)
    bin_ids = pd.qcut(np.arange(n_total), q=n_time_bins, labels=False)
    X_full = pd.concat([X_train_raw[kept_features], X_test_raw[kept_features]], axis=0)

    bin_rows: list[dict[str, Any]] = []
    for b in range(n_time_bins):
        mask = bin_ids == b
        y_b = y.iloc[mask]
        ts_b = timestamps.iloc[mask]
        n_b = int(len(y_b))
        k_b = int(y_b.sum())
        ci_low, ci_high = wilson_ci(k_b, n_b)
        miss_b = float(X_full.iloc[mask].isna().mean().mean())
        bin_rows.append({
            "bin": b + 1,
            "start_time": ts_b.iloc[0],
            "end_time": ts_b.iloc[-1],
            "n_wafers": n_b,
            "n_failures": k_b,
            "fail_rate": float(k_b / n_b),
            "fail_rate_ci_low": ci_low,
            "fail_rate_ci_high": ci_high,
            "mean_missing_rate": miss_b,
        })
    bins_df = pd.DataFrame(bin_rows)

    contingency = np.column_stack([
        bins_df["n_failures"].to_numpy(),
        (bins_df["n_wafers"] - bins_df["n_failures"]).to_numpy(),
    ])
    _, label_drift_p, _, _ = chi2_contingency(contingency)

    drift_rows: list[dict[str, Any]] = []
    for col in kept_features:
        tr_vals = X_train_raw[col].dropna().to_numpy()
        te_vals = X_test_raw[col].dropna().to_numpy()

        tr_miss = float(X_train_raw[col].isna().mean())
        te_miss = float(X_test_raw[col].isna().mean())

        if len(tr_vals) >= 5 and len(te_vals) >= 5:
            ks_stat, ks_p = ks_2samp(tr_vals, te_vals)
            psi = compute_psi(tr_vals, te_vals)
            tr_std = float(np.std(tr_vals, ddof=0))
            smd = float((np.mean(te_vals) - np.mean(tr_vals)) / tr_std) if tr_std > 0 else 0.0
        else:
            ks_stat, ks_p, psi, smd = 0.0, 1.0, 0.0, 0.0

        drift_rows.append({
            "feature": col,
            "psi": float(psi),
            "ks_stat": float(ks_stat),
            "ks_p_value": float(ks_p),
            "std_mean_diff": smd,
            "train_missing_rate": tr_miss,
            "test_missing_rate": te_miss,
            "abs_missing_rate_shift": float(abs(te_miss - tr_miss)),
        })

    feature_drift_df = pd.DataFrame(drift_rows)
    feature_drift_df["ks_fdr_q"] = benjamini_hochberg(feature_drift_df["ks_p_value"].to_numpy())
    feature_drift_df["psi_flagged"] = feature_drift_df["psi"] >= psi_warn_threshold
    feature_drift_df["ks_flagged"] = feature_drift_df["ks_fdr_q"] < ks_alpha
    feature_drift_df = feature_drift_df.sort_values(
        ["psi", "ks_stat"], ascending=False
    ).reset_index(drop=True)

    n_train = len(X_train_raw)
    n_test = len(X_test_raw)
    y_train_rate = float(y.iloc[:n_train].mean())
    y_test_rate = float(y.iloc[n_train:].mean())

    summary = {
        "n_time_bins": n_time_bins,
        "label_rate_across_bins_chi2_p": float(label_drift_p),
        "train_fail_rate": y_train_rate,
        "test_fail_rate": y_test_rate,
        "train_n": n_train,
        "test_n": n_test,
        "n_features_evaluated": len(feature_drift_df),
        "n_features_psi_flagged": int(feature_drift_df["psi_flagged"].sum()),
        "n_features_ks_fdr_flagged": int(feature_drift_df["ks_flagged"].sum()),
        "median_psi": float(feature_drift_df["psi"].median()),
        "max_psi": float(feature_drift_df["psi"].max()),
        "top_drifted_features": feature_drift_df.head(5)[
            ["feature", "psi", "ks_fdr_q", "std_mean_diff"]
        ].to_dict(orient="records"),
    }
    return bins_df, feature_drift_df, summary


def save_phase1_figures(
    bins_df: pd.DataFrame,
    feature_drift_df: pd.DataFrame,
    raw_missing_rates: pd.Series,
    figures_dir: str | Path,
) -> list[Path]:
    """Save Phase 1 diagnostic figures for drift and missingness.

    Args:
        bins_df: Chronological bin summary DataFrame.
        feature_drift_df: Per-feature PSI/KS drift DataFrame.
        raw_missing_rates: Missingness rates across all raw features.
        figures_dir: Output directory for PNG figures.

    Returns:
        List of saved figure Paths.
    """
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Temporal drift overview (failure rate with 95% Wilson CI + top PSI features).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    x = bins_df["bin"].to_numpy()
    y_rate = bins_df["fail_rate"].to_numpy() * 100.0
    y_err_low = y_rate - bins_df["fail_rate_ci_low"].to_numpy() * 100.0
    y_err_high = bins_df["fail_rate_ci_high"].to_numpy() * 100.0 - y_rate

    axes[0].errorbar(
        x, y_rate, yerr=[y_err_low, y_err_high],
        fmt="-o", capsize=4, color="#1f4e78", linewidth=2,
    )
    axes[0].set_title("Failure Rate Over Time (95% Wilson CI)")
    axes[0].set_xlabel("Chronological Quintile Bin")
    axes[0].set_ylabel("Failure Rate (%)")
    axes[0].set_xticks(x)
    axes[0].grid(alpha=0.3)

    top_psi = feature_drift_df.head(15).iloc[::-1]
    axes[1].barh(top_psi["feature"], top_psi["psi"], color="#b85450")
    axes[1].axvline(0.25, color="black", linestyle="--", linewidth=1, label="PSI = 0.25")
    axes[1].set_title("Top 15 Shifted Features (Train vs. Test PSI)")
    axes[1].set_xlabel("Population Stability Index (PSI)")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    drift_fig = out_dir / "phase1_drift_overview.png"
    fig.savefig(drift_fig, dpi=150)
    plt.close(fig)
    saved.append(drift_fig)

    # 2. Missingness histogram across raw features + missingness over time.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(raw_missing_rates.to_numpy() * 100.0, bins=25, color="#4a7c59", edgecolor="white")
    axes[0].axvline(30.0, color="#d97706", linestyle="--", label="Flag > 30%")
    axes[0].axvline(55.0, color="#dc2626", linestyle="--", label="Drop > 55%")
    axes[0].set_title("Raw Sensor Missingness Distribution (Train)")
    axes[0].set_xlabel("Missing Rate (%)")
    axes[0].set_ylabel("Number of Features")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        bins_df["bin"],
        bins_df["mean_missing_rate"] * 100.0,
        "-s",
        color="#4a7c59",
        linewidth=2,
    )
    axes[1].set_title("Mean Feature Missingness Over Time")
    axes[1].set_xlabel("Chronological Quintile Bin")
    axes[1].set_ylabel("Mean Missingness (%)")
    axes[1].set_xticks(x)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    miss_fig = out_dir / "phase1_missingness_overview.png"
    fig.savefig(miss_fig, dpi=150)
    plt.close(fig)
    saved.append(miss_fig)

    return saved
