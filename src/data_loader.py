from ucimlrepo import fetch_ucirepo
import pandas as pd


def loader():
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)
    X = individual_household_electric_power_consumption.data.features
    X['datetime'] = pd.to_datetime(X['Date'] + ' ' + X['Time'], dayfirst=True) #dayfirst= True because the formet of the date is DD/MM/YYYY
    X.set_index('datetime', inplace=True)
    X.drop(['Date', 'Time'], axis=1, inplace=True)
    return X

def data_loader():
    X = loader()

    X.replace('?', pd.NA, inplace=True)    #missing value is replace by "?" right now i changed it by Nan but need to use the function interpolate later
    X=X.dropna()
    X = X.astype(float)
    return X

def get_hourly_data(X):
    X_hourly = X.resample('h').sum()
    return X_hourly

def get_daily_data(X):
    X_daily = X.resample('D').sum()
    return X_daily

def get_monthly_data(X):
    X_monthly = X.resample('ME').sum()
    return X_monthly


def get_yearly_data(X):
    X_yearly = X.resample('YE').sum()
    return X_yearly

def interpolation(X):
    #pour extrapoler,
    #[u(t+1)-u(t-1)]/2
    return
#data = data_loader()
#data_hourly = get_hourly_data(data)
#data_daily = get_daily_data(data)
#data_monthly = get_monthly_data(data)
#data_yearly = get_yearly_data(data)
#print(data)
#print(data_hourly)
#print(data_daily)
#print(data_monthly)
#print(data_yearly)
