import streamlit as st
import requests
import os
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="Ride Cancellation Predictor", page_icon="🚖")

st.title("Ride Cancellation Predictor")
st.caption("Send ride features to the FastAPI endpoint and get cancellation prediction.")

def _get_secret(name: str):
    try:
        return st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None


default_api_url = (
    _get_secret("PREDICTION_API_URL")
    or os.getenv("PREDICTION_API_URL")
    or "http://127.0.0.1:8000/predict"
)
api_url = st.text_input("Prediction API URL", value=default_api_url)

with st.form("predict_form"):
    st.subheader("Ride Details")
    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["eBike", "Bike", "Auto", "Cab Economy", "Cab Premium", "unknown"],
        index=0,
    )
    pickup_location = st.text_input("Pickup Location", value="Palam Vihar")
    drop_location = st.text_input("Drop Location", value="Jhilmil")
    payment_method = st.selectbox(
        "Payment Method",
        ["Cash", "UPI", "Card", "Wallet", "unknown"],
        index=1,
    )

    st.subheader("Customer and Driver History")
    cancelled_rides_by_customer = st.number_input(
        "Cancelled Rides by Customer", min_value=0.0, value=1.0, step=1.0
    )
    cancelled_rides_by_driver = st.number_input(
        "Cancelled Rides by Driver", min_value=0.0, value=1.0, step=1.0
    )
    avg_vtat = st.number_input("Avg VTAT", min_value=0.0, value=8.0, step=0.5)
    avg_ctat = st.number_input("Avg CTAT", min_value=0.0, value=10.0, step=0.5)
    driver_ratings = st.number_input(
        "Driver Ratings", min_value=0.0, max_value=5.0, value=4.2, step=0.1
    )
    customer_rating = st.number_input(
        "Customer Rating", min_value=0.0, max_value=5.0, value=4.4, step=0.1
    )

    st.subheader("Trip and Time")
    ride_distance = st.number_input("Ride Distance", min_value=0.0, value=3.0, step=0.1)
    booking_value = st.number_input("Booking Value", min_value=0.0, value=240.0, step=10.0)
    booking_hour = st.slider("Booking Hour", min_value=0, max_value=23, value=10)
    booking_day_of_week = st.slider("Booking Day Of Week (0=Mon ... 6=Sun)", 0, 6, 2)
    booking_month = st.slider("Booking Month", min_value=1, max_value=12, value=6)
    is_weekend = st.selectbox("Is Weekend", [0, 1], index=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "vehicle_type": vehicle_type,
        "pickup_location": pickup_location,
        "drop_location": drop_location,
        "payment_method": payment_method,
        "avg_vtat": avg_vtat,
        "avg_ctat": avg_ctat,
        "cancelled_rides_by_customer": cancelled_rides_by_customer,
        "cancelled_rides_by_driver": cancelled_rides_by_driver,
        "booking_value": booking_value,
        "ride_distance": ride_distance,
        "driver_ratings": driver_ratings,
        "customer_rating": customer_rating,
        "booking_day_of_week": booking_day_of_week,
        "booking_month": booking_month,
        "is_weekend": is_weekend,
        "booking_hour": booking_hour,
    }
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
            probability = float(result.get("cancellation_probability", 0.0))
            if is_cancelled == 1:
                st.warning("Predicted outcome: Cancelled")
            else:
                st.success("Predicted outcome: Not cancelled")
            st.write(f"Cancellation probability: {probability:.3f}")
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
