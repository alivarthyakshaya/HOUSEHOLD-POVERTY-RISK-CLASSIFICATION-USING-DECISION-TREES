"""Data loading and cleaning for the household poverty project."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .generate_synthetic_data import TARGET_NAMES, generate

TARGET = "Target"
ID_COLS = ["Id", "idhogar"]


def load_raw(path: str | Path | None) -> tuple[pd.DataFrame, str]:
    """Load the CSV at *path*; if it is missing, fall back to synthetic data.

    Returns the dataframe and a short string naming the data source.
    """
    if path is not None and Path(path).exists():
        return pd.read_csv(path), f"file:{Path(path).name}"
    return generate(), "synthetic"


def _is_kaggle_schema(df: pd.DataFrame) -> bool:
    return {"idhogar", "parentesco1"}.issubset(df.columns)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for modelling.

    * Kaggle schema: keep one row per household (the head, ``parentesco1 == 1``)
      because the poverty label is defined per household.
    * ``yes``/``no`` text values are mapped to 1/0.
    * Structural missing values (no rent, no tablet) are filled with 0 and flagged.
    * Remaining numeric gaps are filled with the column median.
    """
    df = df.copy()
    if _is_kaggle_schema(df):
        df = df[df["parentesco1"] == 1]

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET] + [c for c in ID_COLS if c in df.columns])

    for col in X.select_dtypes(include="object"):
        X[col] = X[col].replace({"yes": 1, "no": 0}).apply(pd.to_numeric, errors="coerce")

    for col in ("v2a1", "v18q1", "rez_esc"):            # missing really means "none"
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(0)

    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add a few interpretable ratio features (only if the source columns exist)."""
    X = X.copy()
    if {"hhsize", "rooms"}.issubset(X.columns):
        X["people_per_room"] = X["hhsize"] / X["rooms"].clip(lower=1)
    if {"rooms", "bedrooms"}.issubset(X.columns):
        X["bedroom_ratio"] = X["bedrooms"] / X["rooms"].clip(lower=1)
    asset_cols = [c for c in ("refrig", "television", "computer", "v18q") if c in X.columns]
    if asset_cols:
        X["asset_score"] = X[asset_cols].sum(axis=1)
    return X


__all__ = ["TARGET", "TARGET_NAMES", "load_raw", "clean", "add_features"]
