from __future__ import annotations

import pandas as pd


LEAKAGE_COLUMNS = [
    "booking_status",
    "reason_for_cancelling_by_customer",
    "driver_cancellation_reason",
    "incomplete_rides_reason",
    "incomplete_rides",
]
ID_COLUMNS = ["booking_id", "customer_id"]
NULL_LIKE = {"", "nan", "none", "null"}


def load_data(path):
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return out


def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    if "booking_status" not in df.columns:
        raise KeyError(
            "Required target column 'booking_status' not found after header normalization. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    status = out["booking_status"].astype(str).str.strip().str.lower()
    status = status.mask(status.isin(NULL_LIKE))
    out = out.loc[status.notna()].copy()
    status = out["booking_status"].astype(str).str.strip().str.lower()

    known_statuses = {
        "completed",
        "cancelled by driver",
        "cancelled by customer",
        "no driver found",
        "incomplete",
    }
    out = out.loc[status.isin(known_statuses)].copy()
    status = out["booking_status"].astype(str).str.strip().str.lower()
    out["is_cancelled"] = (status != "completed").astype(int)
    return out


def _engineer_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        parsed_date = pd.to_datetime(out["date"], errors="coerce", dayfirst=False)
        out["booking_day_of_week"] = parsed_date.dt.dayofweek.fillna(-1).astype(int)
        out["booking_month"] = parsed_date.dt.month.fillna(0).astype(int)
        out["is_weekend"] = out["booking_day_of_week"].isin([5, 6]).astype(int)
        out = out.drop(columns=["date"])

    if "time" in out.columns:
        hour_token = out["time"].astype(str).str.extract(r"^\s*(\d{1,2})")[0]
        hour = pd.to_numeric(hour_token, errors="coerce")
        hour = hour.where(hour.between(0, 23), -1)
        out["booking_hour"] = hour.fillna(-1).astype(int)
        out = out.drop(columns=["time"])

    return out


def clean_data(df):
    df = _normalize_columns(df)
    df = _build_target(df)
    df = _engineer_datetime_features(df)

    drop_columns = [c for c in LEAKAGE_COLUMNS + ID_COLUMNS if c in df.columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)

    # Drop constant features; they add noise and make model metadata larger.
    feature_cols = [c for c in df.columns if c != "is_cancelled"]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)

    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != "is_cancelled"]
    categorical_cols = list(df.select_dtypes(include=["object"]).columns)

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].fillna("unknown")

    return df
