import unittest

import pandas as pd

from src.inference import build_model_row
from src.preprocess import clean_data


class TestPreprocessAndInference(unittest.TestCase):
    def test_clean_data_builds_non_degenerate_target(self):
        raw = pd.DataFrame(
            {
                "Booking Status": ["Completed", "Cancelled by Driver", "No Driver Found", "Incomplete"],
                "Date": ["2025-01-01"] * 4,
                "Time": ["10:15", "11:30", "22:00", "08:45"],
                "Ride Distance": [1.0, 2.0, 3.0, 4.0],
                "Booking ID": ["a", "b", "c", "d"],
                "Customer ID": ["u1", "u2", "u3", "u4"],
            }
        )

        clean = clean_data(raw)
        self.assertIn("is_cancelled", clean.columns)
        self.assertEqual(int(clean["is_cancelled"].sum()), 3)
        self.assertNotIn("booking_id", clean.columns)
        self.assertNotIn("customer_id", clean.columns)
        self.assertIn("booking_hour", clean.columns)
        self.assertNotIn("date", clean.columns)

    def test_build_model_row_maps_aliases(self):
        payload = {"distance": 4.5, "booking_hour": 10}
        row = build_model_row(
            payload=payload,
            categorical_cols=["pickup_location"],
            numerical_cols=["ride_distance", "booking_hour"],
        )
        self.assertEqual(row["ride_distance"], 4.5)
        self.assertEqual(row["booking_hour"], 10.0)


if __name__ == "__main__":
    unittest.main()
