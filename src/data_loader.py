import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd


def loader():
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)
    X = individual_household_electric_power_consumption.data.features
    X['datetime'] = pd.to_datetime(X['Date'] + ' ' + X['Time'], dayfirst=True) #dayfirst= True because the format of the date is DD/MM/YYYY not MM/DD/YYYY
    X.set_index('datetime', inplace=True)
    X.drop(['Date', 'Time'], axis=1, inplace=True)
    print(X)
    return X

def data_loader():
    X = loader()

    X.replace('?', np.nan, inplace=True)    #missing value is replace by "?" right now i changed it by Nan but need to use the function interpolate later
    X=X.astype(float)

    X=interpolation(X)




    X=X.dropna()
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
    n_max_linear=10

    n_missing_value=0
    idx_missing_value=[]

    #case when floowed missing value <=10
    for j in range(X.shape[1]):  # j = colonne
        for i in range(1, len(X)):  # i = ligne
            if np.isnan(X.iloc[i,j]):
                idx_missing_value.append(i)
                n_missing_value+=1
            else:
                if idx_missing_value and len(idx_missing_value) <= n_max_linear:
                    # indices
                    i_start=idx_missing_value[0]-1  # avant le NaN
                    i_end=idx_missing_value[-1]+1  # après le NaN (ou i)

                    y0=X.iloc[i_start,j]
                    y1=X.iloc[i_end,j]

                    # ax+b
                    a=(y1-y0)/(i_end-i_start)
                    b=y0-a*i_start

                    #interpolate for each i missing
                    for k in idx_missing_value:
                        X.iloc[k,j]=a*k+b
                idx_missing_value.clear()
                n_missing_value=0




    return X





data = data_loader()
"""
data = data_loader()
data_hourly = get_hourly_data(data)
data_daily = get_daily_data(data)
data_monthly = get_monthly_data(data)
data_yearly = get_yearly_data(data)
print(data)
print(data_hourly)
print(data_daily)
print(data_monthly)
print(data_yearly)
"""


#ChatGPT to count how many followed missing value there is in the dataframe
"""
mask=X.isna().any(axis=1)  # True si la ligne a au moins un NaN
groups=(mask!=mask.shift()).cumsum()  # numérote les séquences
nan_streaks=mask.groupby(groups).sum()  # compte les NaN consécutifs

# On ne garde que les séquences qui contiennent des NaN
nan_streaks=nan_streaks[nan_streaks>0]

# On compte combien de fois chaque longueur apparaît
streak_counts=nan_streaks.value_counts().sort_index()

# Afficher sous forme de tableau
df_streaks=pd.DataFrame({
    "Longueur de NaN consécutifs": streak_counts.index,
    "Nombre de séquences": streak_counts.values
})

print(df_streaks)
    """
