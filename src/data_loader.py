from ucimlrepo import fetch_ucirepo
import pandas as pd


def loader():
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)

    X = individual_household_electric_power_consumption.data.features
    Y = X['Global_active_power']
    X = X.drop(columns=['Global_active_power'])

    X['datetime'] = pd.to_datetime(X['Date'] + ' ' + X['Time'], dayfirst=True) #dayfirst= True because the formet of the date is DD/MM/YYYY
    X.set_index('datetime', inplace=True)
    Y.index = X.index

    X.drop(columns=['Date', 'Time'], inplace=True)  #delette because the information is in the index

    print(X)
    print(Y)

    print(X.columns)  # noms des colonnes
    print(X.head())  # premières lignes
    print(X.info())  # type et non-null
    print(X.describe())  # stats numériques
    return X,Y

def data_loader():
    X,Y = loader()
    return X,Y

def get_hourly_data(X,Y):
    X_hourly = X.resample('H').mean()
    Y_hourly = Y.resample('H').sum()
    return X_hourly,Y_hourly

def get_daily_data(X,Y):
    X_daily = X.resample('D').mean()
    Y_daily = Y.resample('D').sum()
    return X_daily,Y_daily

def get_monthly_data(X,Y):
    X_monthly = X.resample('M').mean()
    Y_monthly = Y.resample('M').sum()
    return X_monthly,Y_monthly


def get_yearly_data(X,Y):
    X_yearly = X.resample('Y').mean()
    Y_yearly = Y.resample('Y').sum()
    return X_yearly,Y_yearly

def interpolation(X,Y):
    #pour extrapoler,
    #[u(t+1)-u(t-1)]/2
    return

