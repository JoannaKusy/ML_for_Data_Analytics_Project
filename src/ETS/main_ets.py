from direct import run_experiment



CONFIG = {
    "data": {
        "train_path_daily": "data/processed/residential4_energy_demand_daily_train.csv",
        "test_path_daily": "data/processed/residential4_energy_demand_daily_test.csv",
        "train_path_hourly": "data/processed/residential4_energy_demand_hourly_train.csv",
        "test_path_hourly": "data/processed/residential4_energy_demand_hourly_test.csv",
    },

    "model": {
            "HWS":True,
            "seasonal_periods": 7
    },

    "forecast":{
            "resolution": "daily",
            "horizon": 65
    }
}


if __name__ == "__main__":
    run_experiment(CONFIG)