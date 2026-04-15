from network_variants import LSTMModel0, keras_LSTM_encoder_decoder
from direct import run_experiment as run_experiment_torch
from keras_direct import run_experiment as run_experiment_keras


CONFIG = {
    "data": {
        "train_path": "data/processed/residential4_energy_demand_daily_train.csv",
        "test_path": "data/processed/residential4_energy_demand_daily_test.csv",
        "lags": [1,2]
    },

    "model": {
        "network_arch": LSTMModel0,
        "network_params": {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.01
        }
    },

    "training": {
        "epochs": 50,
        "lr": 0.001,
        "batch_size": 32
    },

    "wandb": {
        "entity": "ml-for-data-analytics-project",
        "project": "energy-forecasting",
        "run_name": None #generated automatically if not provided, you can also change later on website
    }
}

CONFIG = {
    "data": {
        "train_path": "data/processed/residential4_energy_demand_daily_train.csv",
        "test_path": "data/processed/residential4_energy_demand_daily_test.csv",
        "lags": [1,2],
        "resolution": "daily",
    },

    "model": {
        "network_arch": keras_LSTM_encoder_decoder,
        "network_params": {
            "encoder_units": 64,
            "decoder_units": 64,
            "dense_units": 32,
            "dropout": 0.2,
            "kernel_regularizer": {"l1": 0.01, "l2": 0.01},
        }
    },

    "training": {
        "epochs": 50,
        "lr": 0.001,
        "batch_size": 32
    },

    "wandb": {
        "entity": "ml-for-data-analytics-project",
        "project": "energy-forecasting",
        "run_name": None #generated automatically if not provided, you can also change later on website
    }
}

if __name__ == "__main__":
    # run_experiment_torch(CONFIG)
    run_experiment_keras(CONFIG)

    pass