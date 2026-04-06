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

    train_target = train_df[["energy_demand"]]
    test_target = test_df[["energy_demand"]]

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

    train_final = pd.concat([train_target, train_cat_df], axis=1)
    test_final = pd.concat([test_target, test_cat_df], axis=1) 

    train_final = train_final.reset_index().rename(columns={'utc_timestamp': 'ds', 'energy_demand': 'y'})
    train_final['ds'] = train_final['ds'].dt.tz_localize(None)
    test_final = test_final.reset_index().rename(columns={'utc_timestamp': 'ds', 'energy_demand': 'y'})
    test_final['ds'] = test_final['ds'].dt.tz_localize(None)

    return train_final, test_final
