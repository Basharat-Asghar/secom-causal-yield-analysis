"""Download and load the UCI SECOM semiconductor manufacturing dataset."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


REQUIRED_FILES = ("secom.data", "secom_labels.data", "secom.names")


def download_secom(
    raw_dir: str | Path,
    url: str = "https://archive.ics.uci.edu/static/public/179/secom.zip",
    force: bool = False,
) -> dict[str, Path]:
    """Download and extract UCI SECOM files if not already present.

    Args:
        raw_dir: Directory where raw SECOM files should be stored.
        url: UCI archive zip URL.
        force: Re-download and overwrite existing files if True.

    Returns:
        Mapping from required filename to extracted local Path.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    paths = {name: raw_path / name for name in REQUIRED_FILES}

    if not force and all(p.exists() and p.stat().st_size > 0 for p in paths.values()):
        return paths

    zip_path = raw_path / "secom.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download UCI SECOM from {url}. "
            f"Please download manually and place {', '.join(REQUIRED_FILES)} into {raw_path.resolve()}."
        ) from exc

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = set(zf.namelist())
        missing = [name for name in REQUIRED_FILES if name not in members]
        if missing:
            raise ValueError(f"Downloaded archive missing expected files: {missing}")
        for name in REQUIRED_FILES:
            zf.extract(name, raw_path)

    return paths


def load_secom(
    raw_dir: str | Path,
    data_file: str = "secom.data",
    labels_file: str = "secom_labels.data",
    sort_by_time: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, int | bool]]:
    """Load UCI SECOM features, binary yield-loss labels, and timestamps.

    Maps original labels: -1 (pass) -> 0, 1 (fail) -> 1.
    When ``sort_by_time`` is True, rows are stably sorted chronologically so
    downstream time-ordered splits never leak future records.

    Args:
        raw_dir: Directory containing ``secom.data`` and ``secom_labels.data``.
        data_file: Feature matrix filename.
        labels_file: Labels and timestamps filename.
        sort_by_time: Whether to stably sort records chronologically.

    Returns:
        Tuple of ``(X, y, timestamps, load_info)`` where ``X`` has columns
        ``sensor_0``..``sensor_589``, ``y`` is integer {0, 1}, ``timestamps`` is
        ``datetime64[ns]``, and ``load_info`` records raw ordering diagnostics.
    """
    raw_path = Path(raw_dir)
    data_path = raw_path / data_file
    labels_path = raw_path / labels_file

    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"SECOM files not found in {raw_path}. Expected {data_path.name} and {labels_path.name}."
        )

    X = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan", "NA"],
        dtype=float,
    )
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    labels_df = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["raw_label", "timestamp_str"],
    )

    if len(X) != len(labels_df):
        raise ValueError(
            f"Row count mismatch between {data_file} ({len(X)}) and {labels_file} ({len(labels_df)})."
        )

    valid_labels = set(labels_df["raw_label"].unique())
    if not valid_labels.issubset({-1, 1}):
        raise ValueError(f"Unexpected label values in {labels_file}: {sorted(valid_labels)}")

    y = (labels_df["raw_label"] == 1).astype(int)
    y.name = "fail"

    timestamps = pd.to_datetime(
        labels_df["timestamp_str"],
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    timestamps.name = "timestamp"

    was_monotonic = bool(timestamps.is_monotonic_increasing)
    n_duplicate_timestamps = int(timestamps.duplicated().sum())

    if sort_by_time and not was_monotonic:
        order = timestamps.argsort(kind="mergesort")
        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        timestamps = timestamps.iloc[order].reset_index(drop=True)

    load_info: dict[str, int | bool] = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_failures": int(y.sum()),
        "raw_was_monotonic_increasing": was_monotonic,
        "n_duplicate_timestamps": n_duplicate_timestamps,
    }
    return X, y, timestamps, load_info
