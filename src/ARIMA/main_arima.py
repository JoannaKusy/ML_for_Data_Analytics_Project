from direct import run_experiment


CONFIG = {
    "data": {
        "train_path": "data/processed/residential4_energy_demand_daily_train.csv",
        "test_path": "data/processed/residential4_energy_demand_daily_test.csv",
        "resolution": "daily",
    },
    "model": {
        "type": "sarima",  # choose one: decompose_arima, sarima, sarimax
        "seasonal_period": 7,
        "decompose_model": "additive",
        "trend": "c",
        "enforce_stationarity": False,
        "enforce_invertibility": False,
        "search_space": {
            "p": [0, 1, 2, 3],
            "q": [0, 1, 2, 3],
        },
        "seasonal_orders": [
            [0, 1, 0, 7],
            [1, 1, 0, 7],
            [0, 1, 1, 7],
            [1, 1, 1, 7],
        ],
    },
    "wandb": {
        "entity": "ml-for-data-analytics-project",
        "project": "energy-forecasting",
        "run_name": None,
    },
}


if __name__ == "__main__":
    run_experiment(CONFIG)
