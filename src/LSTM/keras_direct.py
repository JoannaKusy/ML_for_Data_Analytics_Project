from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


import numpy as np
import wandb
import pandas as pd
from metrics import *
from preprocess import load_data, encode_features, scale_data_new, add_lagged_features, add_lagged_features_new, create_sequences

#run name for wandb
def generate_run_name(config, model):
    name = model.__class__.__name__
    return f"{name}_lr{config['training']['lr']}_seq{config['data']['lags']}"

#main function
def run_experiment(CONFIG):

    train_df = load_data(CONFIG["data"]["train_path"])
    test_df = load_data(CONFIG["data"]["test_path"])

    # categorical features encoding
    train_df, test_df = encode_features(train_df, test_df, resolution=CONFIG["data"]["resolution"])

    # scaling numerical features
    train_df, test_df, scaler = scale_data_new(train_df, test_df)

    lags = CONFIG["data"]["lags"]
    X_past_train, X_future_train, y_train, X_past_test, X_future_test, y_test = create_sequences(train_df, test_df, k=lags[-1], resolution=CONFIG["data"]["resolution"])
    
    input_size = X_past_train.shape[1]
    n_past_features = X_past_train.shape[2]
    n_future_features = X_future_train.shape[2]

    model = CONFIG["model"]["network_arch"](
        input_size=input_size,
        n_past_features=n_past_features,
        n_future_features=n_future_features,
        **CONFIG["model"]["network_params"]
    )

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["training"]["lr"]),
        loss='mse',
        metrics=['mse', 'mae']
    )

    # early_stopping = EarlyStopping(
    #     monitor='train_loss',
    #     patience=5,
    #     restore_best_weights=True
    # )

    run_name = CONFIG["wandb"]["run_name"] or generate_run_name(CONFIG, model)

    #info prints
    print("~~~~~~~~~~ Launching training ~~~~~~~~~~~~")
    print(f"Run name: {run_name}")
    print(model)
    print(f"Lags used: {lags}")
    print(f"X_past_train shape: {X_past_train.shape}")
    print(f"X_future_train shape: {X_future_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_past_test shape: {X_past_test.shape}")
    print(f"X_future_test shape: {X_future_test.shape}")
    print(f"y_test shape: {y_test.shape}")

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
        validation_data=([X_past_test, X_future_test], y_test),
        epochs=CONFIG["training"]["epochs"],
        batch_size=CONFIG["training"]["batch_size"],
        # callbacks=[early_stopping]
    )

    # log metrics to wandb
    for epoch in range(CONFIG["training"]["epochs"]):
        train_loss = history.history['loss'][epoch]
        val_loss = history.history['val_loss'][epoch]
        val_mse = history.history['val_mse'][epoch]
        val_rmse = np.sqrt(val_mse)
        val_mae = history.history['val_mae'][epoch]
        
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