import streamlit as st
import requests
import os

st.set_page_config(page_title="Ride Cancellation Predictor", page_icon="🚖")

st.title("Ride Cancellation Predictor")
st.caption("Send ride features to the FastAPI endpoint and get cancellation prediction.")

default_api_url = (
    st.secrets.get("PREDICTION_API_URL")
    or os.getenv("PREDICTION_API_URL")
    or "http://127.0.0.1:8000/predict"
)
api_url = st.text_input("Prediction API URL", value=default_api_url)

with st.form("predict_form"):
    distance = st.number_input("Distance", min_value=0.0, value=3.0, step=0.1)
    booking_hour = st.slider("Booking Hour", min_value=0, max_value=23, value=10)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {"distance": distance, "booking_hour": booking_hour}
    try:
        response = requests.post(api_url, json=payload, timeout=10)

        if response.status_code >= 400:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            st.error(f"Prediction failed ({response.status_code}): {detail}")
        else:
            result = response.json()
            is_cancelled = int(result.get("is_cancelled", 0))
            if is_cancelled == 1:
                st.warning("Predicted outcome: Cancelled")
            else:
                st.success("Predicted outcome: Not cancelled")
            st.json(result)
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to API. "
            "Set a reachable endpoint via `PREDICTION_API_URL` "
            "(Streamlit secrets/env var), or run FastAPI locally with: "
            "`python -m uvicorn api.app:app --host 127.0.0.1 --port 8000`"
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
