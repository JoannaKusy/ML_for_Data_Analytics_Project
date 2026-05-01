import random
from copy import deepcopy
from math import prod

from network_variants import LSTMModel0, keras_LSTM_encoder_decoder
from direct import run_experiment as run_experiment_torch
from keras_direct import run_experiment as run_experiment_keras


TORCH_CONFIG = {
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

BASE_CONFIG = {
    "data": {
        "train_path": "data/processed/residential4_energy_demand_daily_train.csv",
        "test_path": "data/processed/residential4_energy_demand_daily_test.csv",
        "lags": [1, 2],
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
        },
    },
    "training": {
        "epochs": 50,
        "lr": 0.001,
        "batch_size": 32,
    },
    "wandb": {
        "entity": "ml-for-data-analytics-project",
        "project": "energy-forecasting",
        "run_name": None,
    },
}


RANDOM_SEARCH = {
    "enabled": True,
    "n_trials": 10,
    "seed": 42,
    "search_space": {
        "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "hidden_size": [32, 64, 96, 128],
        "batch_size": [16, 32, 64],
        "epochs": [30, 50, 80, 100],
    },
}


def build_encoder_decoder_config(params, trial_idx):
    config = deepcopy(BASE_CONFIG)
    hidden_size = params["hidden_size"]

    config["model"]["network_params"]["encoder_units"] = hidden_size
    config["model"]["network_params"]["decoder_units"] = hidden_size
    config["model"]["network_params"]["dense_units"] = max(hidden_size // 2, 16)
    config["model"]["network_params"]["dropout"] = params["dropout"]

    config["training"]["lr"] = params["learning_rate"]
    config["training"]["batch_size"] = params["batch_size"]
    config["training"]["epochs"] = params["epochs"]

    config["wandb"]["run_name"] = (
        f"encdec_rs_t{trial_idx:02d}_"
        f"lr{params['learning_rate']}_"
        f"dr{params['dropout']}_"
        f"hs{params['hidden_size']}_"
        f"bs{params['batch_size']}_"
        f"ep{params['epochs']}"
    )
    return config


def create_random_search_configs(n_trials, search_space, seed=42):
    rng = random.Random(seed)
    max_unique_configs = prod(len(values) for values in search_space.values())
    target_trials = min(n_trials, max_unique_configs)

    configs = []
    used_combinations = set()

    while len(configs) < target_trials:
        params = {name: rng.choice(values) for name, values in search_space.items()}
        combo_key = tuple(params[name] for name in sorted(search_space.keys()))

        if combo_key in used_combinations:
            continue

        used_combinations.add(combo_key)
        configs.append(build_encoder_decoder_config(params, trial_idx=len(configs) + 1))

    return configs


def run_random_search(random_search_cfg):
    configs = create_random_search_configs(
        n_trials=random_search_cfg["n_trials"],
        search_space=random_search_cfg["search_space"],
        seed=random_search_cfg["seed"],
    )

    print(f"Running encoder-decoder random search with {len(configs)} trials")

    for trial_idx, config in enumerate(configs, start=1):
        params = config["model"]["network_params"]
        training = config["training"]
        print(
            f"\nTrial {trial_idx}/{len(configs)} -> "
            f"lr={training['lr']}, dropout={params['dropout']}, "
            f"hidden_size={params['encoder_units']}, batch_size={training['batch_size']}, "
            f"epochs={training['epochs']}"
        )
        run_experiment_keras(config)


if __name__ == "__main__":
    if RANDOM_SEARCH["enabled"]:
        run_random_search(RANDOM_SEARCH)
    else:
        run_experiment_keras(BASE_CONFIG)