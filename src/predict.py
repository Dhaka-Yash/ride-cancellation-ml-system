import argparse
import json
from pathlib import Path

try:
    from src.config import MODEL_PATH
    from src.inference import load_model, predict_from_payload
except ModuleNotFoundError:
    from config import MODEL_PATH
    from inference import load_model, predict_from_payload


def _read_payload(args: argparse.Namespace) -> dict:
    payload: dict = {}

    if args.payload:
        payload.update(json.loads(args.payload))
    if args.payload_file:
        payload.update(json.loads(Path(args.payload_file).read_text(encoding="utf-8")))

    if args.distance is not None:
        payload["distance"] = args.distance
    if args.booking_hour is not None:
        payload["booking_hour"] = args.booking_hour

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single ride-cancellation prediction.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to model .pkl file")
    parser.add_argument("--payload", help="Inline JSON payload for prediction")
    parser.add_argument("--payload-file", help="Path to JSON payload file")
    parser.add_argument("--distance", type=float, help="Ride distance (alias for ride_distance)")
    parser.add_argument("--booking-hour", type=int, help="Booking hour 0-23 (maps to time HH:00)")
    args = parser.parse_args()

    payload = _read_payload(args)
    if not payload:
        raise ValueError(
            "No input provided. Use --payload/--payload-file or simple flags like "
            "--distance and --booking-hour."
        )

    model_path = Path(args.model_path)
    model = load_model(model_path)
    prediction = predict_from_payload(payload, model)

    print(json.dumps({"is_cancelled": prediction}))


if __name__ == "__main__":
    main()
