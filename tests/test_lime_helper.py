import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import humanize_lime_condition
from app import humanize_lime_condition


def test_categorical_single_value():
    result = humanize_lime_condition("sex > 0.50")
    assert result == "Sex is Male"


def test_categorical_double_bound():
    result = humanize_lime_condition("0.50 < cp <= 1.50")
    assert result == "Chest Pain Type is Atypical Angina"


def test_continuous_double_bound():
    result = humanize_lime_condition("150.00 < trestbps <= 160.00")
    assert "Resting Blood Pressure" in result
    assert "150" in result
    assert "160" in result
    assert "mmHg" in result


def test_continuous_above_threshold():
    result = humanize_lime_condition("chol > 260.00")
    assert result == "Cholesterol Level is above 260 mg/dL"


def test_continuous_below_threshold():
    result = humanize_lime_condition("chol <= 211.00")
    assert result == "Cholesterol Level is 211 mg/dL or below"


def test_unknown_feature_falls_back_unchanged():
    result = humanize_lime_condition("unknown_feature > 5.00")
    assert result == "unknown_feature > 5.00"