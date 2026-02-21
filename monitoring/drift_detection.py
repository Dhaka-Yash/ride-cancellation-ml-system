from pathlib import Path
import argparse

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PROJECT_ROOT / "data" / "ncr_ride_bookings.csv"
DEFAULT_CURRENT = DEFAULT_REFERENCE
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "drift_report.html"


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    project_relative = PROJECT_ROOT / p
    if project_relative.exists():
        return project_relative
    return p.resolve()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return out


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}. "
            f"Provide an existing path via --{label}."
        )
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an Evidently data drift report.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="Path to reference CSV")
    parser.add_argument("--current", default=str(DEFAULT_CURRENT), help="Path to current CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output HTML report")
    args = parser.parse_args()

    reference_path = _resolve_path(args.reference)
    current_path = _resolve_path(args.current)
    output_path = _resolve_path(args.output)

    reference = _normalize_columns(_load_csv(reference_path, "reference"))
    current = _normalize_columns(_load_csv(current_path, "current"))

    common_columns = [col for col in reference.columns if col in current.columns]
    if not common_columns:
        raise ValueError("No common columns found between reference and current datasets.")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference[common_columns],
        current_data=current[common_columns],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    print(f"Drift report saved to: {output_path}")


if __name__ == "__main__":
    main()
