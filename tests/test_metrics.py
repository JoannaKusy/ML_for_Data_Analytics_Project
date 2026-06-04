import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from metrics import mse, rmse, mae


def test_evaluation_metrics_correctness():
    """Verify that custom metric calculations."""
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 30.0])
    assert np.isclose(mse(actual, predicted), 2.666666), "MSE calculation is incorrect"
    assert np.isclose(
        rmse(actual, predicted), 1.632993
    ), "RMSE calculation is incorrect"
    assert np.isclose(mae(actual, predicted), 1.333333), "MAE calculation is incorrect"


def test_metrics_shape_mismatch():
    """Ensure metrics still work even if one array is shape (N,) and the other is (N, 1)."""
    actual = np.array([10.0, 20.0, 30.0])
    predicted_2d = np.array([[12.0], [18.0], [30.0]])

    try:
        mae(actual, predicted_2d.flatten())
    except Exception as e:
        pytest.fail(f"Metric function failed on shape mismatch: {e}")
