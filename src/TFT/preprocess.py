import pandas as pd


def load_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def preprocess_data(train, test):
    train = train.reset_index()
    test = test.reset_index()

    train["day"] = (train["utc_timestamp"] - train["utc_timestamp"].min()).dt.days
    test["day"] = (test["utc_timestamp"] - train["utc_timestamp"].min()).dt.days
    train.drop(columns=["utc_timestamp"], inplace=True)
    test.drop(columns=["utc_timestamp"], inplace=True)

    train["is_holiday_or_weekend"] = (
        train["is_holiday_or_weekend"].astype("str").astype("category")
    )
    test["is_holiday_or_weekend"] = (
        test["is_holiday_or_weekend"].astype("str").astype("category")
    )

    train["group_id"] = 0
    test["group_id"] = 0

    cnct = pd.concat([train, test], ignore_index=True)
    return train, test, cnct
