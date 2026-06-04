import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from preprocess import load_data, preprocess_data
import wandb


def run_experiment(config):
    pl.seed_everything(config["seed"], workers=True)

    # Load and preprocess data
    train_df = load_data(config["data"]["train_path"])
    test_df = load_data(config["data"]["test_path"])
    train, test, cnct = preprocess_data(train_df, test_df)

    # Create TimeSeriesDataSet for TFT
    max_encoder_length = max(config["data"]["lags"])
    max_prediction_length = 1  # predicting one step ahead

    training = TimeSeriesDataSet(
        train,
        time_idx="day",
        target="energy_demand",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[
            "energy_demand",
            "dishwasher", 
            "ev", 
            "freezer", 
            "grid_export", 
            "heat_pump", 
            "washing_machine", 
            "temperature", 
            "radiation_direct_horizontal", 
            "radiation_diffuse_horizontal"
        ],
        time_varying_known_categoricals=["season", "is_holiday_or_weekend"],
        target_normalizer = EncoderNormalizer(method="robust", center=True)
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, cnct, predict=False, min_prediction_idx = test.at[0, "day"], stop_randomization=True
    )

    # Create dataloaders
    batch_size = config["training"]["batch_size"]
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # Define TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config["training"]["lr"],
        hidden_size=config["model"]["network_params"]["hidden_size"],
        attention_head_size=config["model"]["network_params"]["attention_head_size"],
        dropout=config["model"]["network_params"]["dropout"],
        hidden_continuous_size=config["model"]["network_params"]["hidden_continuous_size"],
        # output_size=7,  # quantiles: [0.1, 0.5, 0.9]
        loss=QuantileLoss(),
    )

    wandb.login()

    logger = pl.loggers.WandbLogger(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        config=config
    )

    # Train the model
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        gradient_clip_val = 0.01, 
        accelerator       = "mps", 
        log_every_n_steps = 5,
        enable_progress_bar = True,
        logger            = logger,
        enable_checkpointing = False,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5
            )
        ],
    )

    trainer.fit(tft, train_dataloader, val_dataloader)

    raw_preds = tft.predict(validation, mode="raw", return_x=True)
    true = raw_preds.x["decoder_target"]

    # all prediction windows for that group
    preds = raw_preds.output.prediction

    pred_1step = preds[:, 0].detach().cpu().numpy()
    true_1step = true[:, 0].detach().cpu().numpy()

    median = pred_1step[:, 3]

    df_preds = pd.DataFrame(
        {
            "date": test_df.index,
            "actual_kWh": true_1step,
            "predicted_kWh": median,
        }
    )

    df_preds["error"] = df_preds["actual_kWh"] - df_preds["predicted_kWh"]

    logger.experiment.log({"predictions": wandb.Table(dataframe=df_preds)})

    logger.experiment.log(
        {
            "actual_vs_predicted": wandb.plot.line_series(
                xs=list(range(len(df_preds))),
                ys=[
                    df_preds["actual_kWh"].tolist(),
                    df_preds["predicted_kWh"].tolist(),
                ],
                keys=["actual", "predicted"],
                title="Actual vs Predicted (kWh)",
                xname="time_step",
            )
        }
    )

    logger.experiment.finish()
