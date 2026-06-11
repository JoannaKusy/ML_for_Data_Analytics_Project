from copy import deepcopy

import optuna

from network_variants import keras_LSTM_encoder_decoder
from keras_direct import run_experiment as run_experiment_keras

BASE_CONFIG = {
    "seed": 42,
    "data": {
        "train_path": "data/processed/residential4_energy_demand_daily_train.csv",
        "test_path": "data/processed/residential4_energy_demand_daily_test.csv",
        "lags": [1, 2, 3, 4],
        "val_size": 111,
        "val_ratio": None,
        "resolution": "daily",
    },
    "model": {
        "network_arch": keras_LSTM_encoder_decoder,
        "network_params": {
            "encoder_units": 128,
            "decoder_units": 128,
            "dense_units": 64,
            "dropout": 0.1,
            "kernel_regularizer": {"l1": 0.01, "l2": 0.1},
        },
    },
    "training": {
        "epochs": 130,
        "lr": 0.00071,
        "batch_size": 7,
    },
    "wandb": {
        "entity": "ml-for-data-analytics-project",
        "project": "energy-forecasting",
        "run_name": "encdec_lstm_base",
        "enabled": True,
    },
}


OPTUNA_SEARCH = {
    "enabled": False,
    "n_trials": 30,
    "seed": 42,
    "objective_metric": "best_val_rmse",
    "search_space": {
        "learning_rate": {"low": 5e-4, "high": 9e-4, "step": 1e-5},
        "dropout": {"low": 0.1, "high": 0.2, "step": 0.02},
        "hidden_size": {"low": 64, "high": 128, "step": 8},
        "batch_size": {"low": 6, "high": 8, "step": 1},
        "epochs": {"low": 100, "high": 140, "step": 10},
        "val_ratio": {"low": 0.20, "high": 0.30, "step": 0.05},
    },
}


def build_encoder_decoder_config(params, trial_name):
    config = deepcopy(BASE_CONFIG)
    hidden_size = params["hidden_size"]

    config["model"]["network_params"]["encoder_units"] = hidden_size
    config["model"]["network_params"]["decoder_units"] = hidden_size
    config["model"]["network_params"]["dense_units"] = max(hidden_size // 2, 16)
    config["model"]["network_params"]["dropout"] = params["dropout"]

    config["training"]["lr"] = params["learning_rate"]
    config["training"]["batch_size"] = params["batch_size"]
    config["training"]["epochs"] = params["epochs"]

    # Use ratio-driven split during search for small datasets.
    if "val_ratio" in params:
        config["data"]["val_ratio"] = params["val_ratio"]

    config["wandb"]["run_name"] = (
        f"encdec_optuna_{trial_name}_"
        f"lr{params['learning_rate']}_"
        f"dr{params['dropout']}_"
        f"hs{params['hidden_size']}_"
        f"bs{params['batch_size']}_"
        f"ep{params['epochs']}"
    )
    return config


def sample_params(trial, search_space):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", **search_space["learning_rate"]
        ),
        "dropout": trial.suggest_float("dropout", **search_space["dropout"]),
        "hidden_size": trial.suggest_int("hidden_size", **search_space["hidden_size"]),
        "batch_size": trial.suggest_int("batch_size", **search_space["batch_size"]),
        "epochs": trial.suggest_int("epochs", **search_space["epochs"]),
        "val_ratio": trial.suggest_float("val_ratio", **search_space["val_ratio"]),
    }


def run_optuna_search(optuna_cfg):
    sampler = optuna.samplers.TPESampler(seed=optuna_cfg["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"Running Optuna search with {optuna_cfg['n_trials']} trials")

    def objective(trial):
        params = sample_params(trial, optuna_cfg["search_space"])
        trial_name = f"t{trial.number:02d}"
        config = build_encoder_decoder_config(params, trial_name=trial_name)
        config["wandb"]["enabled"] = False

        print(
            f"\nTrial {trial.number + 1}/{optuna_cfg['n_trials']} -> "
            f"lr={params['learning_rate']}, dropout={params['dropout']}, "
            f"hidden_size={params['hidden_size']}, batch_size={params['batch_size']}, "
            f"epochs={params['epochs']}, val_ratio={params['val_ratio']}"
        )

        metrics = run_experiment_keras(config)
        trial.set_user_attr("metrics", metrics)
        return metrics[optuna_cfg["objective_metric"]]

    study.optimize(objective, n_trials=optuna_cfg["n_trials"])

    best_params = study.best_params
    best_metrics = study.best_trial.user_attrs.get("metrics", {})

    print("\nBest trial found by Optuna")
    print(f"  objective({optuna_cfg['objective_metric']}): {study.best_value:.6f}")
    print(f"  params: {best_params}")
    if best_metrics:
        print(f"  best_val_rmse: {best_metrics.get('best_val_rmse')}")
        print(f"  test_mae: {best_metrics.get('test_mae')}")
        print(
            f"  split(train/val): {best_metrics.get('train_size')}/{best_metrics.get('val_size')}"
        )

    final_config = build_encoder_decoder_config(best_params, trial_name="best")
    final_config["wandb"]["enabled"] = True
    run_experiment_keras(final_config)


if __name__ == "__main__":
    if OPTUNA_SEARCH["enabled"]:
        run_optuna_search(OPTUNA_SEARCH)
    else:
        run_experiment_keras(BASE_CONFIG)
