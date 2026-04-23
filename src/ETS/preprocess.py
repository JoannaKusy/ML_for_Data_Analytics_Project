import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

def load_data(path):
    full_path = BASE_DIR / path
    df = pd.read_csv(full_path, index_col=0, parse_dates=True)
    return df



def encode_features(train_df, test_df, resolution="daily"):
    # TRAIN
    train_final = train_df[["energy_demand"]].copy()
    train_final = train_final.reset_index().rename(columns={
        "utc_timestamp": "ds",
        "energy_demand": "y"
    })
    train_final["ds"] = train_final["ds"].dt.tz_localize(None)
    train_final = train_final.sort_values("ds")

    # TEST
    test_final = test_df[["energy_demand"]].copy()
    test_final = test_final.reset_index().rename(columns={
        "utc_timestamp": "ds",
        "energy_demand": "y"
    })
    test_final["ds"] = test_final["ds"].dt.tz_localize(None)
    test_final = test_final.sort_values("ds")

    return train_final, test_final