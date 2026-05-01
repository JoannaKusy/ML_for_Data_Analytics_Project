import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import numpy as np
from preprocess import load_data, encode_features, scale_data_new, create_sequences
from metrics import mse, rmse, mae
import pandas as pd

#run name for wandb
def generate_run_name(config, model):
    name = model.__class__.__name__
    return f"{name}_lr{config['training']['lr']}"

def run_experiment(CONFIG):

    train_df = load_data(CONFIG["data"]["train_path"])
    test_df = load_data(CONFIG["data"]["test_path"])
    
    train_df, test_df = encode_features(train_df, test_df, resolution=CONFIG["data"]["resolution"])
    train_df, test_df, scaler = scale_data_new(train_df, test_df)

    lags = CONFIG["data"]["lags"]
    X_past_train, X_future_train, y_train, X_past_test, X_future_test, y_test = create_sequences(train_df, test_df, k=lags[-1], resolution=CONFIG["data"]["resolution"])
    
    input_size = X_past_train.shape[1]
    n_past_features = X_past_train.shape[2]
    n_future_features = X_future_train.shape[2]

    def get_loader(p, f, target):
        return DataLoader(
            TensorDataset(torch.tensor(p, dtype=torch.float32),
                          torch.tensor(f, dtype=torch.float32),
                          torch.tensor(target, dtype=torch.float32)),
            batch_size=CONFIG["training"]["batch_size"], shuffle=False
        )

    train_loader = get_loader(X_past_train, X_future_train, y_train)
    test_loader = get_loader(X_past_test, X_future_test, y_test)

    # Model Initialization
    model = CONFIG["model"]["network_arch"](
        input_size=input_size,
        n_past_features = n_past_features,
        n_future_features=n_future_features,
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
    print(f"X_past_train shape: {X_past_train.shape}")
    print(f"X_future_train shape: {X_future_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_past_test shape: {X_past_test.shape}")
    print(f"X_future_test shape: {X_future_test.shape}")
    print(f"y_test shape: {y_test.shape}")

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

    CONFIG["run_name"] = run_name
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

        for Xp, Xf, y in train_loader:
            Xp, Xf, y = Xp.to(device), Xf.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(Xp, Xf).view(-1)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        preds, targets = [], []
        val_loss = 0

        with torch.no_grad():
            for Xp, Xf, y in test_loader:
                Xp, Xf, y = Xp.to(device), Xf.to(device), y.to(device)

                pred = model(Xp, Xf).view(-1)
                loss = criterion(pred, y.view(-1))
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

    n_features = scaler.n_features_in_

    preds_full = np.zeros((len(preds), n_features))
    targets_full = np.zeros((len(targets), n_features))

    preds_full[:, 0] = preds
    targets_full[:, 0] = targets

    preds_rescaled = scaler.inverse_transform(preds_full)[:, 0]
    targets_rescaled = scaler.inverse_transform(targets_full)[:, 0]

    df_preds = pd.DataFrame({
        "date": test_df.index,
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

    #for shap analysis
    #save the model weights
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    
    artifact = wandb.Artifact(
        name=f"trained_model_{run_name}", 
        type="model",
        description="Final model weights and scaler for SHAP analysis"
    )
    
    artifact.add_file(model_path)
    
    import joblib
    scaler_path = "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    artifact.add_file(scaler_path)
    
    run.log_artifact(artifact)
    run.finish()
    wandb.finish()