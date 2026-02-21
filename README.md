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

## Run Streamlit UI

```powershell
streamlit run app\streamlit_app.py
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
