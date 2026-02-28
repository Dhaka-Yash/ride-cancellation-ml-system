# Ride Cancellation ML System

Production-style baseline for training and serving a ride-cancellation classifier.

## Project Structure

`src/` model training, preprocessing, and CLI inference  
`api/` FastAPI prediction service  
`app/` Streamlit UI client  
`monitoring/` data drift report generation  
`data/` input datasets

## Prerequisites

- Python 3.10+
- Windows PowerShell (commands below) or equivalent shell

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train Model

```powershell
python src\train.py
```

Expected output: model saved to `models/model.pkl`.
MLflow runs are tracked in `mlflow.db` (SQLite backend).

## Run API

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Quick health check:

```powershell
curl http://127.0.0.1:8000/
```

Sample prediction request:

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"vehicle_type\":\"Bike\",\"pickup_location\":\"Palam Vihar\",\"drop_location\":\"Jhilmil\",\"payment_method\":\"UPI\",\"avg_vtat\":8,\"avg_ctat\":10,\"cancelled_rides_by_customer\":1,\"cancelled_rides_by_driver\":0,\"booking_value\":250,\"ride_distance\":3.5,\"driver_ratings\":4.5,\"customer_rating\":4.6,\"booking_day_of_week\":2,\"booking_month\":6,\"is_weekend\":0,\"booking_hour\":10}"
```

Sample response:

```json
{
  "is_cancelled": 0,
  "cancellation_probability": 0.347
}
```

API routes:
- `GET /` health/info
- `POST /predict` prediction endpoint (used by Streamlit)

## Run Streamlit UI

```powershell
streamlit run app\streamlit_app.py
```

The UI now includes all core model features (ride details, history, and trip/time fields) so predictions are not biased by missing inputs.

## Deploy Streamlit (Cloud)

When deploying `app/streamlit_app.py` to Streamlit Cloud, do not use `127.0.0.1` for API calls.  
Set a public FastAPI endpoint in app secrets:

```toml
PREDICTION_API_URL = "https://your-fastapi-domain/predict"
```

Important: the secret must include `/predict`.  
If you set only the base URL (for example `https://your-fastapi-domain`), Streamlit can fail with `405 Method Not Allowed`.

The Streamlit app reads API URL in this order:
- `st.secrets["PREDICTION_API_URL"]`
- `PREDICTION_API_URL` environment variable
- Local fallback: `http://127.0.0.1:8000/predict`

If `secrets.toml` is missing, the app continues to run and uses env/local fallback.

## Deploy API Publicly (Render Example)

Create a Render Web Service from this repository and use:

- Build command:

```bash
pip install -r requirements.txt && python src/train.py
```

- Start command:

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port $PORT
```

Then set Streamlit secret:

```toml
PREDICTION_API_URL = "https://<your-render-service>.onrender.com/predict"
```

## CLI Prediction

Simple flags:

```powershell
python src\predict.py --distance 4.5 --booking-hour 10
```

JSON payload:

```powershell
python src\predict.py --payload "{\"distance\": 4.5, \"booking_hour\": 10}"
```

## Run Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Data Drift Report

Default (reference vs same dataset):

```powershell
python monitoring\drift_detection.py
```

Custom current dataset:

```powershell
python monitoring\drift_detection.py --current data\new_data.csv
```

## Notes

- If API returns model-not-found, run `python src\train.py` first.
- Generated artifacts (`venv/`, `mlruns/`, `models/`, caches) are excluded by `.gitignore`.
