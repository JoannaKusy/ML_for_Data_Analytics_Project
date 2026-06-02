import pandas as pd
import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from LSTM.preprocess import *
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

def test_clean_raw_data(monkeypatch):
    """Test the cleaning pipeline"""

    dates = pd.date_range(start='2015-10-14 00:00:00', periods=48, freq='h', tz='UTC')
    
    # dummy Household Data (Simulating cumulative meter readings)
    grid_import_vals = np.concatenate([np.arange(1, 25), np.arange(25, 49) * 2])
    
    dummy_household = pd.DataFrame({
        'cet_cest_timestamp': ['ignore_me'] * 48, # Should be dropped
        'DE_KN_residential4_grid_import': grid_import_vals,
        'DE_KN_residential4_grid_export': np.zeros(48),
        'DE_KN_residential4_pv': np.zeros(48),
        'DE_KN_residential4_dishwasher': np.zeros(48),
        'DE_KN_residential4_ev': np.zeros(48),
        'DE_KN_residential4_freezer': np.zeros(48),
        'DE_KN_residential4_heat_pump': np.zeros(48),
        'DE_KN_residential4_washing_machine': np.zeros(48),
        'DE_KN_residential4_refrigerator': np.zeros(48), # should be dropped
    }, index=dates)

    # dummy Weather Data
    temp_vals = np.concatenate([np.full(24, 10.0), np.full(24, 20.0)])
    
    dummy_weather = pd.DataFrame({
        'DE_temperature': temp_vals,
        'DE_radiation_direct_horizontal': np.ones(48) * 5,
        'DE_radiation_diffuse_horizontal': np.ones(48) * 2,
        'FR_temperature': np.ones(48) * 99 #should be dropped
    }, index=dates)

    def mock_read_csv(filepath, **kwargs):
        if 'household_data' in filepath:
            return dummy_household.copy()
        elif 'weather_data' in filepath:
            return dummy_weather.copy()
        else:
            raise ValueError(f"Unexpected file path: {filepath}")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    
    hourly_df, daily_df = clean_raw_data('fake/data/dir')

    expected_columns = [
        'dishwasher', 'ev', 'freezer', 'grid_export', 'grid_import', 
        'heat_pump', 'pv', 'washing_machine', 'temperature', 
        'radiation_direct_horizontal', 'radiation_diffuse_horizontal'
    ]
    assert list(hourly_df.columns) == expected_columns, "Hourly columns do not match expected output"
    
    # expect 47 rows because diff() makes the first row NaN, which is then dropped.
    assert hourly_df.shape[0] == 47, f"Hourly dataset should have exactly 47 rows, but has {hourly_df.shape[0]}"
    
    assert 'cet_cest_timestamp' not in hourly_df.columns, "Timezone column was not dropped"
    # We passed 2 days of data. The .diff().iloc[1:] should drop the first day.
    # Therefore, we expect exactly 1 day remaining in the daily dataset.
    assert daily_df.shape[0] >= 1, "Daily dataset should have at least 1 complete day"
    
    if daily_df.shape[0] == 1:
        assert daily_df.index[0] == pd.Timestamp('2015-10-15 00:00:00', tz='UTC'), "Remaining day should be the 15th"
    
    # Check the .diff() math on residential data
    # Day 1 mean grid_import = mean(1..24) = 12.5
    # Day 2 mean grid_import = mean(50, 52..96) = 73.0
    # Diff for Day 2 = 73.0 - 12.5 = 60.5
    assert abs(daily_df.loc['2015-10-15', 'grid_import'] - 60.5) < 1e-6, "Diff calculation failed for residential data"
    
    # Check the weather data (ensure it was NOT diffed)
    # Day 2 temp should just be the mean of day 2 (20.0), NOT 20 - 10 = 10.
    assert daily_df.loc['2015-10-15', 'temperature'] == 20.0, "Weather data was incorrectly diffed instead of averaged"
    assert daily_df.loc['2015-10-15', 'radiation_direct_horizontal'] == 5.0

import os
import pandas as pd
import numpy as np
import pytest

def test_run_data_pipeline(monkeypatch, tmp_path):
    """Test the full end-to-end pipeline including file generation and train/test splits."""

    # We use 4 days: Dec 30, Dec 31 (2016) and Jan 1, Jan 2 (2017)
    # This allows us to test the 2017 train/test split.
    # Total periods: 4 days * 24 hours = 96 hours
    dates = pd.date_range(start='2016-12-30 00:00:00', periods=96, freq='h', tz='UTC')
    
    # dummy Household Data
    grid_import_vals = np.arange(1, 97) * 2.0
    
    dummy_household = pd.DataFrame({
        'cet_cest_timestamp': ['ignore_me'] * 96,
        'DE_KN_residential4_grid_import': grid_import_vals,
        'DE_KN_residential4_grid_export': np.zeros(96),
        'DE_KN_residential4_pv': np.zeros(96),
        'DE_KN_residential4_dishwasher': np.zeros(96),
        'DE_KN_residential4_ev': np.zeros(96),
        'DE_KN_residential4_freezer': np.zeros(96),
        'DE_KN_residential4_heat_pump': np.zeros(96),
        'DE_KN_residential4_washing_machine': np.zeros(96),
        'DE_KN_residential4_refrigerator': np.zeros(96),
    }, index=dates)

    # dummy Weather Data
    dummy_weather = pd.DataFrame({
        'DE_temperature': np.full(96, 5.0),
        'DE_radiation_direct_horizontal': np.ones(96),
        'DE_radiation_diffuse_horizontal': np.ones(96)
    }, index=dates)

    original_read_csv = pd.read_csv

    def mock_read_csv(filepath, **kwargs):
        filepath_str = str(filepath)
        
        if 'household_data' in filepath_str:
            return dummy_household.copy()
        elif 'weather_data' in filepath_str:
            return dummy_weather.copy()
        else:
            return original_read_csv(filepath, **kwargs)
            
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    

    # tmp_path provides a temporary dir
    fake_raw_dir = 'fake/raw/dir'
    output_dir = tmp_path / "processed"
    
    hourly_all, daily_all = run_data_pipeline(fake_raw_dir, output_dir)
    

    expected_files = [
        'residential4_energy_demand_hourly.csv',
        'residential4_energy_demand_hourly_train.csv',
        'residential4_energy_demand_hourly_test.csv',
        'residential4_energy_demand_daily.csv',
        'residential4_energy_demand_daily_train.csv',
        'residential4_energy_demand_daily_test.csv'
    ]

    saved_files = os.listdir(output_dir)
    assert len(saved_files) == 6, f"Expected 6 files, found {len(saved_files)}"
    for f in expected_files:
        assert f in saved_files, f"Missing expected file: {f}"

    expected_hourly_features = ['is_holiday_or_weekend', 'daylight_flag', 'time_of_day', 'season']
    for feat in expected_hourly_features:
        assert feat in hourly_all.columns, f"Hourly data is missing feature: {feat}"

    # New Year's Day (Jan 1, 2017) is correctly flagged as a holiday/weekend
    assert daily_all.loc['2017-01-01', 'is_holiday_or_weekend'] == True, "Holiday flag failed for Jan 1st"
    
    # ('winter' for Dec/Jan)
    assert daily_all.loc['2017-01-01', 'season'] == 'winter', "Season feature mapping failed"


    # (first row dropped by diff(), so 95 total hours)
    # 2016: Dec 30 (23 hours) + Dec 31 (24 hours) = 47 hours
    # 2017: Jan 1 (24 hours) + Jan 2 (24 hours) = 48 hours
    df_hourly_train = pd.read_csv(output_dir / 'residential4_energy_demand_hourly_train.csv', index_col=0)
    df_hourly_test = pd.read_csv(output_dir / 'residential4_energy_demand_hourly_test.csv', index_col=0)
    
    assert df_hourly_train.shape[0] == 47, "Hourly Train dataset size incorrect (Should only be 2016)"
    assert df_hourly_test.shape[0] == 48, "Hourly Test dataset size incorrect (Should only be 2017)"
    
    # Dec 30th dropped by diff)
    # 2016: Dec 31 (1 day)
    # 2017: Jan 1 + Jan 2 (2 days)
    df_daily_train = pd.read_csv(output_dir / 'residential4_energy_demand_daily_train.csv', index_col=0)
    df_daily_test = pd.read_csv(output_dir / 'residential4_energy_demand_daily_test.csv', index_col=0)
    
    assert df_daily_train.shape[0] == 1, "Daily Train dataset size incorrect"
    assert df_daily_test.shape[0] == 2, "Daily Test dataset size incorrect"



def test_encode_features():
    """
    Test that categorical features are correctly one-hot encoded 
    and the original categorical columns are dropped.
    """
    #dummy daily data
    train_data = pd.DataFrame({
        'energy_demand': [100, 110, 120],
        'is_holiday_or_weekend': [True, False, False],
        'season': ['winter', 'winter', 'spring']
    })
    
    test_data = pd.DataFrame({
        'energy_demand': [130],
        'is_holiday_or_weekend': [True],
        'season': ['spring']
    })

    train_encoded, test_encoded = encode_features(train_data, test_data, resolution="daily")

    # Check that original categorical columns are gone
    assert 'season' not in train_encoded.columns
    assert 'is_holiday_or_weekend' not in train_encoded.columns
    
    # Check that it retained the numeric column
    assert 'energy_demand' in train_encoded.columns
    
    # Check that new encoded columns exist (e.g., season_winter)
    # OneHotEncoder with drop='first' will drop one category to prevent multicollinearity
    assert len(train_encoded.columns) > 1 
    assert len(test_encoded.columns) == len(train_encoded.columns)
