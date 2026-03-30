import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wandb
import pandas as pd
from src.metrics import *
from preprocess import load_data, encode_features, scale_data, add_lagged_features

#run name for wandb
def generate_run_name(config, model):
    name = model.__class__.__name__
    return f"{name}_lr{config['training']['lr']}_seq{config['data']['lags']}"

#main function
def run_experiment(CONFIG):

    train_df = load_data(CONFIG["data"]["train_path"])
    test_df = load_data(CONFIG["data"]["test_path"])

    train_df, test_df = encode_features(train_df, test_df)

    lags = CONFIG["data"]["lags"]

    train_df = add_lagged_features(train_df, lags)
    test_df = add_lagged_features(test_df, lags)

    train_scaled, test_scaled, scaler = scale_data(train_df, test_df)

    X_train = train_scaled[:, 1:]
    y_train = train_scaled[:, 0]

    X_test = test_scaled[:, 1:]
    y_test = test_scaled[:, 0]

    X_train = X_train[:, None, :]
    X_test = X_test[:, None, :]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ),
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False
    )

    input_size = X_train.shape[2]

    model = CONFIG["model"]["network_arch"](
        input_size=input_size,
        **CONFIG["model"]["network_params"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["training"]["lr"])
    criterion = nn.MSELoss()

    run_name = CONFIG["wandb"]["run_name"] or generate_run_name(CONFIG, model)

    #info prints
    print("~~~~~~~~~~ Launching training ~~~~~~~~~~~~")
    print(f"Run name: {run_name}")
    print(model)
    print(f"Lags used: {lags}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of features: {X_train.shape[2]}")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    for epoch in range(CONFIG["training"]["epochs"]):

        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        preds, targets = [], []
        val_loss = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                pred = model(X).squeeze()
                loss = criterion(pred, y)
                val_loss += loss.item()

                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())

        preds = np.array(preds)
        targets = np.array(targets)

        val_mse = mse(targets, preds)

        run.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/mse": val_mse,
            "val/rmse": rmse(targets, preds),
            "val/mae": mae(targets, preds),
        }, step=epoch)

        print(f"Epoch {epoch} | RMSE: {rmse(targets, preds):.4f}")

    n_features = train_scaled.shape[1]

    preds_full = np.zeros((len(preds), n_features))
    targets_full = np.zeros((len(targets), n_features))

    preds_full[:, 0] = preds
    targets_full[:, 0] = targets

    preds_rescaled = scaler.inverse_transform(preds_full)[:, 0]
    targets_rescaled = scaler.inverse_transform(targets_full)[:, 0]

    dates = test_df.index

    df_preds = pd.DataFrame({
        "date": dates,
        "actual_kWh": targets_rescaled,
        "predicted_kWh": preds_rescaled,
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