from direct import run_experiment


CONFIG = {
    "seed": 42,
    "data": {
        "train_path": "data/processed/residential4_energy_demand_daily_train.csv",
        "test_path": "data/processed/residential4_energy_demand_daily_test.csv",
        "lags": [1, 2],
    },
    "model": {
        "network_params": {
            "hidden_size": 32,
            "hidden_continuous_size": 4,
            "attention_head_size": 1,
            "dropout": 0.2,
        },
    },
    "training": {
        "epochs": 100,
        "lr": 0.0001,
        "batch_size": 8,
    },
    "wandb": {
        "entity": "ml-for-data-analytics-project",
        "project": "energy-forecasting",
        "run_name": "TFT"
    },
}


if __name__ == "__main__":
    run_experiment(CONFIG)
