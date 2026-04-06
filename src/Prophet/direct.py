import prophet
import numpy as np
import wandb
import pandas as pd
from metrics import *
from preprocess import load_data, encode_features

#run name for wandb
def generate_run_name(config):
    name = "Prophet"
    return f"{name}_c_scale{config['model']['changepoint_prior_scale']}_s_scale{config['model']['seasonality_prior_scale']}"

#main function
def run_experiment(CONFIG):

    train_df = load_data(CONFIG["data"]["train_path"])
    test_df = load_data(CONFIG["data"]["test_path"])

    train_df, test_df = encode_features(train_df, test_df, resolution=CONFIG["data"]["resolution"])

    run_name = CONFIG["wandb"]["run_name"] or generate_run_name(CONFIG)

    #info prints
    print("~~~~~~~~~~ Launching training ~~~~~~~~~~~~")
    print(f"Run name: {run_name}")
    

    wandb.login()
    run = wandb.init(
        entity=CONFIG["wandb"]["entity"],
        project=CONFIG["wandb"]["project"],
        name=run_name,
        config=CONFIG
    )

    #training loop

    regressor_cols = train_df.columns.difference(['ds', 'y'])
    _train_df = train_df.copy()
    rows = []
    preds = []

    for i in range(len(test_df)):
        
        m = prophet.Prophet(**CONFIG["model"])
        
        for col in regressor_cols:
            m.add_regressor(col)
            
        m.fit(_train_df)

        _train_df = pd.concat([_train_df, test_df.iloc[i:i+1]], ignore_index=True)
        future = _train_df.drop(columns=['y'])
        
        forecast = m.predict(future)
        next_row = forecast.iloc[-1]
        rows.append(next_row)
        preds.append(next_row['yhat'])

    preds = np.array(preds)
    forecast = pd.concat([
        forecast.iloc[:-len(test_df)],
        pd.DataFrame(rows, index=test_df.index)
    ], ignore_index=True)


    df_preds = pd.DataFrame({
        "date": test_df.ds.values,
        "actual_kWh": test_df.y.values,
        "predicted_kWh": preds,
    })

    df_preds["error"] = df_preds["actual_kWh"] - df_preds["predicted_kWh"]

    run.log({
        "predictions": wandb.Table(dataframe=df_preds)
    })

    run.log({
        "val/mse": mse(test_df.y.values, preds),
        "val/rmse": rmse(test_df.y.values, preds),
        "val/mae": mae(test_df.y.values, preds),
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