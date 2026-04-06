from direct import run_experiment


CONFIG = {
    "data": {
        "train_path": "data/processed/residential4_energy_demand_daily_train.csv",
        "test_path": "data/processed/residential4_energy_demand_daily_test.csv",
        "resolution": "daily",
    },

    "model": {
        "changepoint_prior_scale": 5.0,  # Adjust this value to control the flexibility of the trend (higher values allow more changepoints, deafault is 0.05
        "seasonality_prior_scale": 10.0,  # Adjust this value to control the flexibility of the seasonalities (higher values allow more flexible seasonalities, default is 10.0)
        "weekly_seasonality": True,
        "yearly_seasonality": True,
    },

    "wandb": {
        "entity": "ml-for-data-analytics-project",
        "project": "energy-forecasting",
        "run_name": None #generated automatically if not provided, you can also change later on website
    }
}


if __name__ == "__main__":
    run_experiment(CONFIG)
