import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import holidays
from astral.sun import sun
from astral import LocationInfo
import os


def load_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def clean_raw_data(data_dir):
    """
    Reads raw OPSD data, performs filtering, handles NaNs, and merges 
    specific household data with selected weather data columns.
    
    Args:
        data_dir (str): Path to the 'data/raw' directory.
        
    Returns:
        tuple: (residential4_hourly_merged, residential4_daily_merged)
    """

    household_data_path = os.path.join(data_dir, 'opsd-household_data-2020-04-15/household_data_60min_singleindex.csv')
    weather_data_path = os.path.join(data_dir, 'opsd-weather_data-2020-09-16/weather_data.csv')
    
    household_data = pd.read_csv(household_data_path, index_col=0, parse_dates=True)
    weather_data = pd.read_csv(weather_data_path, index_col=0, parse_dates=True)
    
    if 'cet_cest_timestamp' in household_data.columns:
        household_data = household_data.drop('cet_cest_timestamp', axis=1)
        
    household_data.columns = household_data.columns.str.replace('DE_KN_', '', regex=False)
    
    res4_data = household_data.filter(regex='^residential4_').copy()
    res4_data.columns = res4_data.columns.str.replace('residential4_', '', regex=False)
    
    missing_counts = res4_data.isnull().sum(axis=1)
    threshold_val = np.floor(0.2 * res4_data.shape[1])
    keep_row = missing_counts <= threshold_val
    
    start_day = res4_data.index[keep_row].min()
    end_day = res4_data.index[keep_row].max()
    res4_data = res4_data.loc[start_day:end_day]

    residential4_weather = weather_data.loc[start_day:end_day].copy()
    residential4_weather = residential4_weather.filter(regex='^DE_')
    residential4_weather.columns = residential4_weather.columns.str.replace('DE_', '', regex=False)
    
    weather_cols_to_keep = ['temperature', 'radiation_direct_horizontal', 'radiation_diffuse_horizontal']
    residential4_weather = residential4_weather[weather_cols_to_keep]
    
    
    res4_daily = res4_data.resample('D').mean()
    res4_daily = res4_daily.diff().iloc[1:]
    weather_daily = residential4_weather.resample('D').mean()
    
    res4_data = res4_data.diff().iloc[1:]
    merged_hourly = res4_data.join(residential4_weather, how='left')
    
    merged_daily = res4_daily.join(weather_daily, how='left')
    
    final_columns = [
        'dishwasher', 'ev', 'freezer', 'grid_export', 'grid_import', 
        'heat_pump', 'pv', 'washing_machine', 'temperature', 
        'radiation_direct_horizontal', 'radiation_diffuse_horizontal'
    ]
    
    merged_hourly = merged_hourly[final_columns]
    merged_daily = merged_daily[final_columns]
    merged_hourly = merged_hourly.dropna()
    merged_daily = merged_daily.dropna()
    
    return merged_hourly, merged_daily

import pandas as pd
import holidays
from astral.sun import dawn, dusk
from astral import LocationInfo

def engineer_features(df_hourly, df_daily):
    """
    Applies feature engineering from the EDA phase to the hourly and daily datasets.
    Generates time-based, seasonal, and holiday features.
    
    Args:
        df_hourly (pd.DataFrame): Cleaned hourly data.
        df_daily (pd.DataFrame): Cleaned daily data.
        
    Returns:
        tuple: (hourly_all, hourly_train, hourly_test, daily_all, daily_train, daily_test)
    """
    hourly = df_hourly.copy()
    daily = df_daily.copy()
    
    for df in [hourly, daily]:
        if 'grid_import' in df.columns:
            df['energy_demand'] = df['grid_import']
            df.drop(columns=['grid_import'], inplace=True)
            
    de_holidays = holidays.CountryHoliday('DE', subdiv='BW', years=range(2015, 2018))
    holiday_dates = pd.to_datetime(list(de_holidays.keys()))
    
    for df in [hourly, daily]:
        df['is_holiday_or_weekend'] = df.index.to_series().apply(
            lambda x: x.floor('D') in holiday_dates or x.weekday() >= 5
        )

    city = LocationInfo("Konstanz", "Germany")
    hourly['daylight_flag'] = hourly.index.to_series().apply(
        lambda x: dawn(city.observer, date=x.date()) <= x <= dusk(city.observer, date=x.date())
    )

    def get_time_of_day(date):
        hour = date.hour
        if 0 <= hour < 6: return 'night'
        elif 6 <= hour < 12: return 'morning'
        elif 12 <= hour < 18: return 'afternoon'
        else: return 'evening'
        
    hourly['time_of_day'] = hourly.index.to_series().apply(get_time_of_day)

    def get_season(date):
        month = date.month
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'autumn'
        
    for df in [hourly, daily]:
        df['season'] = df.index.to_series().apply(get_season)

    def reorder_columns(df):
        cols = ['energy_demand'] + [c for c in df.columns if c != 'energy_demand']
        return df[cols]
        
    hourly = reorder_columns(hourly)
    daily = reorder_columns(daily)

    hourly_train = hourly.loc[hourly.index.year != 2017]
    hourly_test = hourly.loc[hourly.index.year == 2017]
    
    daily_train = daily.loc[daily.index.year != 2017]
    daily_test = daily.loc[daily.index.year == 2017]

    return hourly, hourly_train, hourly_test, daily, daily_train, daily_test

def run_data_pipeline(raw_data_dir, output_data_dir):
    """
    Executes the full data pipeline: Preprocessing -> Feature Engineering -> Saving Data.
    """
    print("Running preprocessing...")
    df_hourly_clean, df_daily_clean = clean_raw_data(raw_data_dir)
    
    print("Running Feature Engineering...")
    (hourly_all, hourly_train, hourly_test, 
     daily_all, daily_train, daily_test) = engineer_features(df_hourly_clean, df_daily_clean)
    
    print(f"Saving output to {output_data_dir}...")
    os.makedirs(output_data_dir, exist_ok=True)
 
    hourly_all.to_csv(os.path.join(output_data_dir, 'residential4_energy_demand_hourly.csv'))
    hourly_train.to_csv(os.path.join(output_data_dir, 'residential4_energy_demand_hourly_train.csv'))
    hourly_test.to_csv(os.path.join(output_data_dir, 'residential4_energy_demand_hourly_test.csv'))
    
    daily_all.to_csv(os.path.join(output_data_dir, 'residential4_energy_demand_daily.csv'))
    daily_train.to_csv(os.path.join(output_data_dir, 'residential4_energy_demand_daily_train.csv'))
    daily_test.to_csv(os.path.join(output_data_dir, 'residential4_energy_demand_daily_test.csv'))
    
    print("Data pipeline completed successfully!")
    
    return hourly_all, daily_all


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

