from itertools import product

import numpy as np
import pandas as pd
import wandb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from metrics import mae, mse, rmse
from preprocess import encode_features, load_data, seasonal_decompose_target


def generate_run_name(model_type, order, seasonal_order=None, seasonal_period=None):
    if model_type == "decompose_arima":
        return f"DECOMPOSE_ARIMA_order{tuple(order)}_p{seasonal_period}"
    if model_type == "sarima":
        return f"SARIMA_order{tuple(order)}_sorder{tuple(seasonal_order)}"
    return f"SARIMAX_order{tuple(order)}_sorder{tuple(seasonal_order)}"


def _adf_p_value(series):
    clean_series = pd.Series(series).dropna().astype(float)
    return float(adfuller(clean_series, autolag="AIC")[1])


def _choose_d(series, max_d=2, alpha=0.05):
    differenced = pd.Series(series).copy()
    for d in range(max_d + 1):
        if _adf_p_value(differenced) <= alpha:
            return d
        differenced = differenced.diff()
    return max_d


def _choose_D(series, seasonal_period, max_D=1, alpha=0.05):
    seasonally_differenced = pd.Series(series).copy()
    for D in range(max_D + 1):
        if _adf_p_value(seasonally_differenced) <= alpha:
            return D
        seasonally_differenced = seasonally_differenced.diff(seasonal_period)
    return max_D


def _print_stationarity_choice(label, d=None, D=None):
    if D is None:
        print(f"Selected d for {label}: {d}")
    else:
        print(f"Selected d and D for {label}: d={d}, D={D}")


def _resolve_trend(trend, d, D=0):
    if trend == "c" and (d + D) > 0:
        return "n"
    return trend


def _select_arima_order(train_y, search_space, trend):
    best_order = None
    best_aic = np.inf
    scores = []

    for order in product(search_space["p"], [search_space["d"]], search_space["q"]):
        try:
            fitted = ARIMA(train_y, order=order, trend=trend).fit()
            scores.append({"order": order, "aic": float(fitted.aic), "bic": float(fitted.bic)})
            if fitted.aic < best_aic:
                best_aic = float(fitted.aic)
                best_order = order
        except Exception:
            continue

    if best_order is None:
        raise ValueError("No valid ARIMA order found.")

    scores_df = pd.DataFrame(scores).sort_values("aic") if scores else pd.DataFrame(columns=["order", "aic", "bic"])
    return best_order, scores_df


def _select_sarima_order(train_y, exog, search_space, seasonal_orders, trend, enforce_stationarity, enforce_invertibility):
    best_order = None
    best_seasonal_order = None
    best_aic = np.inf
    scores = []

    for order in product(search_space["p"], [search_space["d"]], search_space["q"]):
        for seasonal_order in seasonal_orders:
            try:
                fitted = SARIMAX(
                    endog=train_y,
                    exog=exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    enforce_stationarity=enforce_stationarity,
                    enforce_invertibility=enforce_invertibility,
                ).fit(disp=False)
                scores.append({"order": order, "seasonal_order": seasonal_order, "aic": float(fitted.aic), "bic": float(fitted.bic)})
                if fitted.aic < best_aic:
                    best_aic = float(fitted.aic)
                    best_order = order
                    best_seasonal_order = seasonal_order
            except Exception:
                continue

    if best_order is None:
        raise ValueError("No valid SARIMA/SARIMAX configuration found.")

    scores_df = pd.DataFrame(scores).sort_values("aic") if scores else pd.DataFrame(columns=["order", "seasonal_order", "aic", "bic"])
    return best_order, best_seasonal_order, scores_df


def _forecast_arima_onestep(train_y, order, trend):
    """Fit ARIMA and predict 1 step ahead with confidence interval."""
    fitted = ARIMA(train_y, order=order, trend=trend).fit()
    forecast = fitted.get_forecast(steps=1)
    pred = forecast.predicted_mean.iloc[0]
    ci = forecast.conf_int(alpha=0.05).iloc[0]
    return pred, ci.iloc[0], ci.iloc[1], fitted


def _forecast_sarimax_onestep(train_y, exog_train, exog_test_row, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility):
    """Fit SARIMAX and predict 1 step ahead with confidence interval."""
    fitted = SARIMAX(
        endog=train_y,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    ).fit(disp=False)
    if exog_test_row is None:
        forecast = fitted.get_forecast(steps=1)
    else:
        forecast = fitted.get_forecast(steps=1, exog=exog_test_row.values.reshape(1, -1))
    pred = forecast.predicted_mean.iloc[0]
    ci = forecast.conf_int(alpha=0.05).iloc[0]
    return pred, ci.iloc[0], ci.iloc[1], fitted


def _run_decompose_arima(train_df, test_df, config):
    seasonal_period = config["model"].get("seasonal_period", 7)
    deseasonalized_train, _, seasonal_cycle = seasonal_decompose_target(
        train_df,
        period=seasonal_period,
        model=config["model"].get("decompose_model", "additive"),
    )

    search_space = config["model"].get("search_space", {"p": [0, 1, 2, 3], "q": [0, 1, 2, 3]})
    trend = config["model"].get("trend", "c")
    search_space["d"] = _choose_d(deseasonalized_train)
    trend = _resolve_trend(trend, search_space["d"], 0)
    _print_stationarity_choice("decomposed ARIMA", d=search_space["d"])
    order, scores = _select_arima_order(deseasonalized_train, search_space, trend)
    print(scores.head(10))
    print(f"Best ARIMA order by AIC: {order}")

    # Walk-forward one-step-ahead forecasting
    _train_y = deseasonalized_train.copy()
    preds = []
    lower_bounds = []
    upper_bounds = []
    fitted = None

    for i in range(len(test_df)):
        pred, lower, upper, fitted = _forecast_arima_onestep(_train_y, order, trend)
        preds.append(pred)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        # Append actual value for next iteration
        actual_val = test_df.iloc[i]["energy_demand"]
        seasonal_adjustment = seasonal_cycle[(i % seasonal_period)] if seasonal_period > 0 else 0
        _train_y = pd.concat([_train_y, pd.Series([actual_val - seasonal_adjustment])], ignore_index=True)

    predictions = np.array(preds) + np.resize(seasonal_cycle, len(test_df))
    lower = np.array(lower_bounds) + np.resize(seasonal_cycle, len(test_df))
    upper = np.array(upper_bounds) + np.resize(seasonal_cycle, len(test_df))

    return predictions, pd.DataFrame({"lower": lower, "upper": upper}, index=test_df.index), fitted, order


def _run_sarima(train_df, test_df, config):
    search_space = config["model"].get("search_space", {"p": [0, 1, 2, 3], "q": [0, 1, 2, 3]})
    seasonal_period = config["model"].get("seasonal_period", 7)
    seasonal_orders_cfg = config["model"].get(
        "seasonal_orders",
        [(0, 1, 0, seasonal_period), (1, 1, 0, seasonal_period), (0, 1, 1, seasonal_period), (1, 1, 1, seasonal_period)],
    )
    trend = config["model"].get("trend", "c")
    enforce_stationarity = config["model"].get("enforce_stationarity", False)
    enforce_invertibility = config["model"].get("enforce_invertibility", False)

    search_space["d"] = _choose_d(train_df["energy_demand"])
    seasonal_D = _choose_D(train_df["energy_demand"], seasonal_period)
    trend = _resolve_trend(trend, search_space["d"], seasonal_D)
    _print_stationarity_choice("SARIMA", d=search_space["d"], D=seasonal_D)
    seasonal_orders = [(so[0], seasonal_D, so[2], so[3]) for so in seasonal_orders_cfg]

    order, seasonal_order, scores = _select_sarima_order(
        train_df["energy_demand"],
        exog=None,
        search_space=search_space,
        seasonal_orders=seasonal_orders,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    print(scores.head(10))
    print(f"Best SARIMA order by AIC: {order}")
    print(f"Best SARIMA seasonal order by AIC: {seasonal_order}")

    # Walk-forward one-step-ahead forecasting
    _train_y = train_df["energy_demand"].copy()
    preds = []
    lower_bounds = []
    upper_bounds = []
    fitted = None

    for i in range(len(test_df)):
        pred, lower, upper, fitted = _forecast_sarimax_onestep(_train_y, None, None, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility)
        preds.append(pred)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        # Append actual value for next iteration
        actual_val = test_df.iloc[i]["energy_demand"]
        _train_y = pd.concat([_train_y, pd.Series([actual_val])], ignore_index=True)

    return np.array(preds), pd.DataFrame({"lower": np.array(lower_bounds), "upper": np.array(upper_bounds)}, index=test_df.index), fitted, order, seasonal_order


def _run_sarimax(train_df, test_df, config):
    train_encoded, test_encoded = encode_features(train_df, test_df, resolution=config["data"]["resolution"])
    exog_train = train_encoded.drop(columns=["energy_demand"])
    exog_test = test_encoded.drop(columns=["energy_demand"])

    search_space = config["model"].get("search_space", {"p": [0, 1, 2, 3], "q": [0, 1, 2, 3]})
    seasonal_period = config["model"].get("seasonal_period", 7)
    seasonal_orders_cfg = config["model"].get(
        "seasonal_orders",
        [(0, 1, 0, seasonal_period), (1, 1, 0, seasonal_period), (0, 1, 1, seasonal_period), (1, 1, 1, seasonal_period)],
    )
    trend = config["model"].get("trend", "c")
    enforce_stationarity = config["model"].get("enforce_stationarity", False)
    enforce_invertibility = config["model"].get("enforce_invertibility", False)

    search_space["d"] = _choose_d(train_df["energy_demand"])
    seasonal_D = _choose_D(train_df["energy_demand"], seasonal_period)
    trend = _resolve_trend(trend, search_space["d"], seasonal_D)
    _print_stationarity_choice("SARIMAX", d=search_space["d"], D=seasonal_D)
    seasonal_orders = [(so[0], seasonal_D, so[2], so[3]) for so in seasonal_orders_cfg]

    order, seasonal_order, scores = _select_sarima_order(
        train_df["energy_demand"],
        exog=exog_train,
        search_space=search_space,
        seasonal_orders=seasonal_orders,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    print(scores.head(10))
    print(f"Best SARIMAX order by AIC: {order}")
    print(f"Best SARIMAX seasonal order by AIC: {seasonal_order}")

    # Walk-forward one-step-ahead forecasting
    _train_y = train_df["energy_demand"].copy()
    _exog_train = exog_train.copy()
    preds = []
    lower_bounds = []
    upper_bounds = []
    fitted = None

    for i in range(len(test_df)):
        exog_test_row = exog_test.iloc[i:i+1]
        pred, lower, upper, fitted = _forecast_sarimax_onestep(_train_y, _exog_train, exog_test_row, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility)
        preds.append(pred)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        # Append actual value for next iteration
        actual_val = test_df.iloc[i]["energy_demand"]
        _train_y = pd.concat([_train_y, pd.Series([actual_val])], ignore_index=True)
        _exog_train = pd.concat([_exog_train, exog_test_row], ignore_index=True)

    return np.array(preds), pd.DataFrame({"lower": np.array(lower_bounds), "upper": np.array(upper_bounds)}, index=test_df.index), fitted, order, seasonal_order


def _log_results(run, test_df, predictions):
    df_preds = pd.DataFrame(
        {
            "date": test_df.index.values,
            "actual_kWh": test_df["energy_demand"].values,
            "predicted_kWh": predictions,
        }
    )
    df_preds["error"] = df_preds["actual_kWh"] - df_preds["predicted_kWh"]

    run.log({"predictions": wandb.Table(dataframe=df_preds)})
    run.log(
        {
            "val/mse": mse(test_df["energy_demand"].values, predictions),
            "val/rmse": rmse(test_df["energy_demand"].values, predictions),
            "val/mae": mae(test_df["energy_demand"].values, predictions),
        }
    )
    run.log(
        {
            "actual_vs_predicted": wandb.plot.line_series(
                xs=list(range(len(df_preds))),
                ys=[df_preds["actual_kWh"].tolist(), df_preds["predicted_kWh"].tolist()],
                keys=["actual", "predicted"],
                title="Actual vs Predicted (kWh)",
                xname="time_step",
            )
        }
    )


def run_experiment(CONFIG):
    train_df = load_data(CONFIG["data"]["train_path"])
    test_df = load_data(CONFIG["data"]["test_path"])
    model_type = CONFIG["model"]["type"].lower()

    if model_type == "sarimax":
        train_data, test_data = train_df, test_df
    elif model_type in {"sarima", "decompose_arima"}:
        train_data = train_df[["energy_demand"]].copy()
        test_data = test_df[["energy_demand"]].copy()
    else:
        raise ValueError(f"Unknown model type: {CONFIG['model']['type']}")

    if model_type == "decompose_arima":
        predictions, conf_int, fitted, order = _run_decompose_arima(train_data, test_data, CONFIG)
        seasonal_order = None
        seasonal_period = CONFIG["model"].get("seasonal_period", 7)
    elif model_type == "sarima":
        predictions, conf_int, fitted, order, seasonal_order = _run_sarima(train_data, test_data, CONFIG)
        seasonal_period = CONFIG["model"].get("seasonal_period", 7)
    else:
        predictions, conf_int, fitted, order, seasonal_order = _run_sarimax(train_data, test_data, CONFIG)
        seasonal_period = CONFIG["model"].get("seasonal_period", 7)

    run_name = CONFIG["wandb"].get("run_name") or generate_run_name(model_type, order, seasonal_order, seasonal_period)

    print("~~~~~~~~~~ Launching training ~~~~~~~~~~~~")
    print(f"Run name: {run_name}")

    wandb.login()
    run = wandb.init(
        entity=CONFIG["wandb"]["entity"],
        project=CONFIG["wandb"]["project"],
        name=run_name,
        config=CONFIG,
    )

    _log_results(run, test_data, predictions)
    run.finish()
