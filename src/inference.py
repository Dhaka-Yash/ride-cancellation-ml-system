from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def extract_expected_columns(model) -> tuple[list[str], list[str], list[str]]:
    preprocessor = model.named_steps["preprocessor"]
    categorical_cols = list(preprocessor.transformers_[0][2])
    numerical_cols = list(preprocessor.transformers_[1][2])
    expected_cols = categorical_cols + numerical_cols
    return categorical_cols, numerical_cols, expected_cols


def build_model_row(
    payload: dict[str, Any],
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {col: "unknown" for col in categorical_cols}
    row.update({col: 0.0 for col in numerical_cols})

    for key, value in payload.items():
        if key in row:
            row[key] = value

    # Backward-compatible aliases used by the Streamlit form.
    if "distance" in payload and "ride_distance" in row:
        row["ride_distance"] = float(payload["distance"])
    if "booking_hour" in payload:
        hour = int(payload["booking_hour"])
        if hour < 0 or hour > 23:
            raise ValueError("booking_hour must be between 0 and 23.")
        if "booking_hour" in row:
            row["booking_hour"] = float(hour)
        elif "time" in row:
            row["time"] = f"{hour:02d}:00"

    for col in numerical_cols:
        try:
            row[col] = float(row[col])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value for '{col}': {row[col]}")

    return row


def predict_from_payload(payload: dict[str, Any], model) -> int:
    categorical_cols, numerical_cols, expected_cols = extract_expected_columns(model)
    row = build_model_row(payload, categorical_cols, numerical_cols)
    df = pd.DataFrame([row], columns=expected_cols)
    return int(model.predict(df)[0])


def predict_with_probability_from_payload(payload: dict[str, Any], model) -> tuple[int, float]:
    categorical_cols, numerical_cols, expected_cols = extract_expected_columns(model)
    row = build_model_row(payload, categorical_cols, numerical_cols)
    df = pd.DataFrame([row], columns=expected_cols)
    prediction = int(model.predict(df)[0])

    probability = 0.0
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])

    return prediction, probability
