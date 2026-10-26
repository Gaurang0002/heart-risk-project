import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from app import best_model, scaler, feature_names

import numpy as np
from app import best_model, scaler, feature_names


def test_model_and_scaler_loaded():
    assert best_model is not None
    assert scaler is not None


def test_prediction_returns_valid_class():
    sample = np.array([[55, 1, 2, 130, 250, 0, 1, 150, 0, 1.2, 1, 0, 2]])
    scaled_sample = scaler.transform(sample)
    prediction = best_model.predict(scaled_sample)[0]
    assert prediction in [0, 1]


def test_prediction_probabilities_sum_to_one():
    sample = np.array([[55, 1, 2, 130, 250, 0, 1, 150, 0, 1.2, 1, 0, 2]])
    scaled_sample = scaler.transform(sample)
    probabilities = best_model.predict_proba(scaled_sample)[0]
    assert round(sum(probabilities), 5) == 1.0


def test_feature_names_match_expected_columns():
    expected = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    assert feature_names == expected