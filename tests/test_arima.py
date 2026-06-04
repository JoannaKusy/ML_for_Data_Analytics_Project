import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def test_arima_fit_predict():
    """Verify ARIMA can fit a basic time series and forecast without matrix errors."""
    # synthetic time series
    time = np.arange(100)
    synthetic_demand = np.sin(time) + np.random.normal(0, 0.1, 100)
    df = pd.Series(synthetic_demand)

    model = ARIMA(df, order=(1, 1, 1))
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=5)

    assert len(forecast) == 5, "Forecast did not return the expected number of steps."
    assert not forecast.isna().any(), "ARIMA forecast produced NaNs."
