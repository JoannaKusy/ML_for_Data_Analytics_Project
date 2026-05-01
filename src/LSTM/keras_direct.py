import os
import random

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


import numpy as np
import wandb
import pandas as pd
from metrics import *
from preprocess import load_data, encode_features, scale_data_new, create_sequences, split_train


def set_reproducible_seed(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass

#run name for wandb
def generate_run_name(config, model):
    name = model.__class__.__name__
    return f"{name}_lr{config['training']['lr']}_seq{config['data']['lags']}"


def resolve_val_size(data_cfg, n_rows):
    val_ratio = data_cfg.get("val_ratio", None)
    if val_ratio is not None:
        val_size = int(round(n_rows * float(val_ratio)))
    else:
        val_size = int(data_cfg.get("val_size", 30))

    # Keep enough observations for both train and validation on small datasets.
    min_val = 20
    max_val = max(n_rows - 20, min_val)
    return max(min_val, min(val_size, max_val))

#main function
def run_experiment(CONFIG):
    seed = CONFIG.get("seed", 42)
    set_reproducible_seed(seed)

    train_df = load_data(CONFIG["data"]["train_path"])
    test_df = load_data(CONFIG["data"]["test_path"])

    # categorical features encoding
    train_df, test_df = encode_features(train_df, test_df, resolution=CONFIG["data"]["resolution"])

    # Splitting train into train and val
    val_size = resolve_val_size(CONFIG["data"], n_rows=len(train_df))
    train_df, val_df = split_train(train_df, val_size=val_size)

    # scaling numerical features
    train_df, val_df, test_df, scaler = scale_data_new(train_df, val_df, test_df)

    lags = CONFIG["data"]["lags"]
    (
        X_past_train,
        X_future_train,
        y_train,
        X_past_val,
        X_future_val,
        y_val,
        X_past_test,
        X_future_test,
        y_test,
    ) = create_sequences(train_df, val_df, test_df, k=lags[-1], resolution=CONFIG["data"]["resolution"])
    
    input_size = X_past_train.shape[1]
    n_past_features = X_past_train.shape[2]
    n_future_features = X_future_train.shape[2]

    model = CONFIG["model"]["network_arch"](
        input_size=input_size,
        n_past_features=n_past_features,
        n_future_features=n_future_features,
        **CONFIG["model"]["network_params"]
    ).get_model()

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["training"]["lr"]),
        loss='mse',
        metrics=['mse', 'mae']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        mode='min'
    )

    run_name = CONFIG["wandb"]["run_name"] or generate_run_name(CONFIG, model)

    #info prints
    print("~~~~~~~~~~ Launching training ~~~~~~~~~~~~")
    print(f"Run name: {run_name}")
    print(model)
    print(f"Lags used: {lags}")
    print(f"X_past_train shape: {X_past_train.shape}")
    print(f"X_future_train shape: {X_future_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_past_val shape: {X_past_val.shape}")
    print(f"X_future_val shape: {X_future_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_past_test shape: {X_past_test.shape}")
    print(f"X_future_test shape: {X_future_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Train/Val rows after split: {len(train_df)}/{len(val_df)}")

    total_params = model.count_params()
    trainable_params = sum(np.prod(v.shape) for v in model.trainable_weights)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    print("\nTrainig config:")
    print(f"Epochs: {CONFIG['training']['epochs']}")
    print(f"Batch size: {CONFIG['training']['batch_size']}")
    print(f"Learning rate: {CONFIG['training']['lr']}")
    print(model)
    print("="*50)



    wandb_cfg = CONFIG.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", True))
    run = None

    if use_wandb:
        wandb.login()
        run = wandb.init(
            entity=CONFIG["wandb"]["entity"],
            project=CONFIG["wandb"]["project"],
            name=run_name,
            config=CONFIG
        )

    #training loop
    history = model.fit(
        [X_past_train, X_future_train], y_train,
        validation_data=([X_past_val, X_future_val], y_val),
        epochs=CONFIG["training"]["epochs"],
        batch_size=CONFIG["training"]["batch_size"],
        callbacks=[early_stopping],
        shuffle=False,
    )

    # log metrics to wandb
    for epoch in range(len(history.history['loss'])):
        train_loss = history.history['loss'][epoch]
        val_loss = history.history['val_loss'][epoch]
        val_mse = history.history['val_mse'][epoch]
        val_rmse = np.sqrt(val_mse)
        val_mae = history.history['val_mae'][epoch]

        if run is not None:
            run.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mse": val_mse,
                "val/rmse": val_rmse,
                "val/mae": val_mae,
            }, step=epoch)


    # predictions
    preds = model.predict([X_past_test, X_future_test], batch_size=CONFIG["training"]["batch_size"]).squeeze()
    preds = scaler.inverse_transform(np.hstack([preds.reshape(-1, 1), np.zeros((len(preds), scaler.n_features_in_ - 1))]))[:, 0]
    targets = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), scaler.n_features_in_ - 1))]))[:, 0]

    df_preds = pd.DataFrame({
        "date": test_df.index,
        "actual_kWh": targets,
        "predicted_kWh": preds,
    })

    df_preds["error"] = df_preds["actual_kWh"] - df_preds["predicted_kWh"]

    if run is not None:
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
                title="Actual vs Predicted (kWh)",
                xname="time_step"
            )
        })

        run.finish()

    best_val_loss = float(np.min(history.history["val_loss"]))
    best_val_rmse = float(np.sqrt(np.min(history.history["val_mse"])))
    test_mae = float(df_preds.error.abs().mean())
    test_rmse = float(np.sqrt(np.mean(np.square(df_preds.error.values))))

    print(test_mae)

    return {
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "val_size": len(val_df),
        "train_size": len(train_df),
    }