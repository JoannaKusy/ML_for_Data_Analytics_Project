import pandas as pd
import numpy as np
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


def add_lagged_features_new(train_df, test_df, lags, resolution="daily"):
    features = {
        "daily": ['is_holiday_or_weekend', 'season'],
        "hourly": ['daylight_flag', 'time_of_day', 'is_holiday_or_weekend', 'season']
    }

    cat_cols = features[resolution]

    # combine train and test so lagged values for the start of test come from end of train
    combined = pd.concat([train_df, test_df]).sort_index()

    # create lagged features on the combined series for all non-categorical cols
    for col in combined.columns.difference(cat_cols):
        for lag in lags:
            combined[f"{col}_lag_{lag}"] = combined[col].shift(lag)

    # split back to train and test using original indices
    train_lagged = combined.loc[train_df.index].copy()
    test_lagged = combined.loc[test_df.index].copy()

    # drop current-time features (keep categorical and target 'energy_demand')
    keep_cols = list(cat_cols) + ['energy_demand']
    drop_cols = [c for c in train_df.columns if c not in keep_cols]

    train_lagged = train_lagged.drop(columns=drop_cols)
    test_lagged = test_lagged.drop(columns=drop_cols)

    # drop rows with NaNs in train (initial rows without full lag history)
    train_lagged = train_lagged.dropna()

    return train_lagged, test_lagged


def create_sequences(train, test, k, resolution="daily"):
    features = {
        "daily": ['is_holiday_or_weekend_True', 'season_spring', 'season_summer', 'season_winter'],
        "hourly": ['daylight_flag', 'time_of_day', 'is_holiday_or_weekend', 'season']
    }

    FUTURE_FEATURES = features[resolution]  # known at prediction time

    ALL_FEATURES = [
        'energy_demand', 'dishwasher', 'ev', 'freezer', 'grid_export',
        'heat_pump', 'pv', 'washing_machine', 'temperature',
        'radiation_direct_horizontal', 'radiation_diffuse_horizontal',
    ] + FUTURE_FEATURES

    TARGET = 'energy_demand'

    combined = pd.concat([train, test]).sort_index()

    X_past, X_future, y = [], [], []

    for i in range(len(combined) - k):
        past = combined.iloc[i:i+k][ALL_FEATURES].values
        future = combined.iloc[i+k:i+k+1][FUTURE_FEATURES].values
        target = combined.iloc[i+k][TARGET]

        X_past.append(past)
        X_future.append(future)
        y.append(target)
    
    
    split_idx = len(train) - k  # first index of test sequences
    X_past_train = X_past[:split_idx]
    X_future_train = X_future[:split_idx]
    y_train = y[:split_idx]

    X_past_test = X_past[split_idx:]
    X_future_test = X_future[split_idx:]
    y_test = y[split_idx:]

    return np.array(X_past_train), np.array(X_future_train), np.array(y_train), np.array(X_past_test), np.array(X_future_test), np.array(y_test)


def scale_data(train, test):
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled, scaler


def scale_data_new(train, test):
    scaler = MinMaxScaler()

    columns = train.columns

    train[columns] = scaler.fit_transform(train[columns])
    test[columns] = scaler.transform(test[columns])

    return train, test, scaler