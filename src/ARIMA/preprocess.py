import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.seasonal import seasonal_decompose


def load_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if isinstance(df.index, pd.DatetimeIndex):
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is not None:
            df = df.asfreq(inferred_freq)
    return df


def encode_features(train_df, test_df, resolution="daily"):
    features = {
        "daily": ["is_holiday_or_weekend", "season"],
        "hourly": ["daylight_flag", "time_of_day", "is_holiday_or_weekend", "season"],
    }

    cat_cols = features[resolution]

    train_target = train_df[["energy_demand"]]
    test_target = test_df[["energy_demand"]]

    train_cat = train_df[cat_cols]
    test_cat = test_df[cat_cols]

    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    train_cat_encoded = encoder.fit_transform(train_cat)
    test_cat_encoded = encoder.transform(test_cat)

    encoded_cols = encoder.get_feature_names_out(cat_cols)

    train_cat_df = pd.DataFrame(train_cat_encoded, columns=encoded_cols, index=train_df.index)
    test_cat_df = pd.DataFrame(test_cat_encoded, columns=encoded_cols, index=test_df.index)

    train_final = pd.concat([train_target, train_cat_df], axis=1)
    test_final = pd.concat([test_target, test_cat_df], axis=1)

    return train_final, test_final


def seasonal_decompose_target(train_df, period, model="additive"):
    target = train_df["energy_demand"].astype(float)
    decomposition = seasonal_decompose(target, model=model, period=period, extrapolate_trend="freq")

    seasonal = decomposition.seasonal.fillna(0.0)
    deseasonalized = target - seasonal
    seasonal_cycle = seasonal.iloc[-period:].to_numpy()

    return deseasonalized, seasonal, seasonal_cycle
