from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import wandb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from metrics import *
from preprocess import load_data, encode_features

#run name for wandb
def generate_run_name(config):
    name = "ETS"
    return f"{name}_{config['model']}"

def run_experiment(CONFIG):
    train_df_d = load_data(CONFIG["data"]["train_path_daily"])
    test_df_d  = load_data(CONFIG["data"]["test_path_daily"])

    train_df_h = load_data(CONFIG["data"]["train_path_hourly"])
    test_df_h  = load_data(CONFIG["data"]["test_path_hourly"])

    train_df_d, test_df_d = encode_features(train_df_d, test_df_d, resolution="daily")
    train_df_h, test_df_h = encode_features(train_df_h, test_df_h, resolution="hourly")


    train_df_d["ds"] = pd.to_datetime(train_df_d["ds"])
    test_df_d["ds"]  = pd.to_datetime(test_df_d["ds"])

    train_df_d = train_df_d.set_index("ds").sort_index()
    test_df_d  = test_df_d.set_index("ds").sort_index()

    y_daily = train_df_d["y"].asfreq("D")


    train_df_h["ds"] = pd.to_datetime(train_df_h["ds"])
    test_df_h["ds"]  = pd.to_datetime(test_df_h["ds"])

    train_df_h = train_df_h.set_index("ds").sort_index()
    test_df_h  = test_df_h.set_index("ds").sort_index()

    y_hourly = train_df_h["y"].asfreq("h")

    run_name=CONFIG["wandb"]["run_name"] or generate_run_name(CONFIG)

    # info prints
    print("~~~~~~~~~~ Launching training ~~~~~~~~~~~~")
    print(f"Run name: {run_name}")

    wandb.login()
    run=wandb.init(
        entity=CONFIG["wandb"]["entity"],
        project=CONFIG["wandb"]["project"],
        name=run_name,
        config=CONFIG
    )



    if CONFIG["model"]["HWS"]:
        model = ExponentialSmoothing(
            y_daily,
            trend="add",
            seasonal="add",
            seasonal_periods=CONFIG["model"]["seasonal_periods"]
        )

        fit = model.fit()
        forecast = fit.forecast(CONFIG["forecast"]["horizon"])
    else:
        #can't because we don't have 2 season
        """
        Y_decomp=seasonal_decompose(
            y_daily,
            model="additive",
            period=365
        )
        seasonal_Y=Y_decomp.seasonal
        """


        W_decomp=seasonal_decompose(y_daily, model="additive", period=7)

        seasonal_pattern=(
            W_decomp.seasonal
            .groupby(W_decomp.seasonal.index.dayofweek)
            .mean()
        )

        y_deseason=y_daily-W_decomp.seasonal

        model=ExponentialSmoothing(y_deseason, trend="add", seasonal=None)
        fit=model.fit()

        forecast_deseason=fit.forecast(CONFIG["forecast"]["horizon"])

        forecast=forecast_deseason+[
            seasonal_pattern[d.dayofweek] for d in forecast_deseason.index
        ]


    if CONFIG["forecast"]["resolution"]=="hourly":
        # HOURLY SEASONALITY
        h_decomp=seasonal_decompose(
            y_hourly,
            model="additive",
            period=24
        )

        seasonal_hourly=h_decomp.seasonal



    y_true = test_df_d["y"].values
    y_pred = forecast.values[:len(y_true)]

    run.log({
        "val/mse": mse(y_true, y_pred),
        "val/rmse": rmse(y_true, y_pred),
        "val/mae": mae(y_true, y_pred),
    })


    df_preds = pd.DataFrame({
        "date": test_df_d.index,
        "actual_kWh": y_true,
        "predicted_kWh": y_pred,
    })

    df_preds["error"] = df_preds["actual_kWh"] - df_preds["predicted_kWh"]

    run.log({
        "predictions": wandb.Table(dataframe=df_preds)
    })

    run.log({
        "actual_vs_predicted": wandb.plot.line_series(
            xs=list(range(len(df_preds))),
            ys=[
                df_preds["actual_kWh"].tolist(),
                df_preds["predicted_kWh"].tolist()
            ],
            keys=["actual", "predicted"],
            title="Actual vs Predicted (ETS)",
            xname="time_step"
        )
    })

    run.finish()

