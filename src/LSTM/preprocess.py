import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def encode_features(train_df, test_df, resolution="daily"):
    features = {
        "daily": ['is_holiday_or_weekend', 'season'],
        "hourly": ['daylight_flag', 'time_of_day', 'is_holiday_or_weekend', 'season']
    }

    cat_cols = features[resolution]

    train_num = train_df.drop(columns=cat_cols)
    test_num = test_df.drop(columns=cat_cols)

    train_cat = train_df[cat_cols]
    test_cat = test_df[cat_cols]

    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    train_cat_encoded = encoder.fit_transform(train_cat)
    test_cat_encoded = encoder.transform(test_cat)

    encoded_cols = encoder.get_feature_names_out(cat_cols)

    train_cat_df = pd.DataFrame(train_cat_encoded, columns=encoded_cols, index=train_df.index)
    test_cat_df = pd.DataFrame(test_cat_encoded, columns=encoded_cols, index=test_df.index)

    train_final = pd.concat([train_num, train_cat_df], axis=1)
    test_final = pd.concat([test_num, test_cat_df], axis=1)

    return train_final, test_final


def add_lagged_features(df, lags):
    df_lagged = df.copy()

    for lag in lags:
        df_lagged[f"lag_{lag}"] = df_lagged["energy_demand"].shift(lag)

    df_lagged = df_lagged.dropna()

    return df_lagged


def scale_data(train, test):
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled, scaler