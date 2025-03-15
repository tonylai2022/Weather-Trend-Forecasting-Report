import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("GlobalWeatherRepository.csv")

# Convert last_updated to datetime and sort
df['last_updated'] = pd.to_datetime(df['last_updated'])
df = df.sort_values('last_updated')

# Handle Outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['temperature_celsius', 'humidity', 'uv_index']:
    df = remove_outliers(df, col)

# Add lagged temperature (temp_t-1)
df['temp_t-1'] = df.groupby('location_name')['temperature_celsius'].shift(1)
df.dropna(subset=['temp_t-1', 'uv_index', 'humidity', 'temperature_celsius'], inplace=True)

# Normalize Data
scaler = MinMaxScaler()
df[['temp_t-1', 'uv_index', 'humidity']] = scaler.fit_transform(df[['temp_t-1', 'uv_index', 'humidity']])

# Split Data (80% Train, 20% Test)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

### MODEL EVALUATION (OVERALL RMSE) ###

# OLS Regression Model
X_train = sm.add_constant(train[['temp_t-1', 'uv_index', 'humidity']])
X_test = sm.add_constant(test[['temp_t-1', 'uv_index', 'humidity']])
y_train = train['temperature_celsius']
y_test = test['temperature_celsius']

ols_model = sm.OLS(y_train, X_train).fit()
ols_pred = ols_model.predict(X_test)
ols_rmse = np.sqrt(mean_squared_error(y_test, ols_pred))

# ARIMA Model Evaluation
arima_rmse_list = []
for location in df['location_name'].unique():
    train_loc = train[train['location_name'] == location]['temperature_celsius']
    test_loc = test[test['location_name'] == location]['temperature_celsius']
    if len(train_loc) > 10 and len(test_loc) > 0:
        try:
            arima_model = ARIMA(train_loc, order=(1,1,2)).fit()
            arima_pred = arima_model.forecast(steps=len(test_loc))
            arima_rmse_list.append(np.sqrt(mean_squared_error(test_loc, arima_pred)))
        except:
            continue
arima_rmse = np.mean(arima_rmse_list) if arima_rmse_list else float('nan')

# Holt-Winters Model Evaluation
hw_rmse_list = []
for location in df['location_name'].unique():
    train_loc = train[train['location_name'] == location]['temperature_celsius']
    test_loc = test[test['location_name'] == location]['temperature_celsius']
    if len(train_loc) > 10 and len(test_loc) > 0:
        try:
            hw_model = ExponentialSmoothing(train_loc, trend='add', seasonal=None).fit()
            hw_pred = hw_model.forecast(steps=len(test_loc))
            hw_rmse_list.append(np.sqrt(mean_squared_error(test_loc, hw_pred)))
        except:
            continue
hw_rmse = np.mean(hw_rmse_list) if hw_rmse_list else float('nan')

### BACKTESTING (LAST 6 DAYS) ###

backtest_period = 6
train_backtest = df.iloc[:-backtest_period]
test_backtest = df.iloc[-backtest_period:]

# OLS Regression Backtest
X_train_bt = sm.add_constant(train_backtest[['temp_t-1', 'uv_index', 'humidity']])
X_test_bt = sm.add_constant(test_backtest[['temp_t-1', 'uv_index', 'humidity']])
y_train_bt = train_backtest['temperature_celsius']
y_test_bt = test_backtest['temperature_celsius']

ols_model_bt = sm.OLS(y_train_bt, X_train_bt).fit()
ols_pred_bt = ols_model_bt.predict(X_test_bt)
ols_rmse_bt = np.sqrt(mean_squared_error(y_test_bt, ols_pred_bt))

# ARIMA Backtest
best_arima_model_bt = ARIMA(train_backtest['temperature_celsius'], order=(1,1,1)).fit()
arima_pred_bt = best_arima_model_bt.forecast(steps=len(test_backtest))
arima_rmse_bt = np.sqrt(mean_squared_error(test_backtest['temperature_celsius'], arima_pred_bt))

# Holt-Winters Backtest
hw_model_bt = ExponentialSmoothing(train_backtest['temperature_celsius'], trend='add', seasonal=None).fit()
hw_pred_bt = hw_model_bt.forecast(steps=len(test_backtest))
hw_rmse_bt = np.sqrt(mean_squared_error(test_backtest['temperature_celsius'], hw_pred_bt))

### COMBINED RESULTS TABLE ###

# Create a combined performance table
combined_results = {
    "Model Type": ["OLS Regression", "Holt-Winters", "ARIMA"],
    "Features": ["Temp_t-1, UV, Humidity", "Temperature", "Temperature"],
    "Scaling or smoothing": ["None", "Additive trend smoothing", "Differencing (1,1,2)"],
    "Overall RMSE": [ols_rmse, hw_rmse, arima_rmse],
    "Backtest RMSE (Last 6 Days)": [ols_rmse_bt, hw_rmse_bt, arima_rmse_bt]
}

combined_results_df = pd.DataFrame(combined_results)

# Display results
print("\n### Combined Model Performance Table ###")
print(combined_results_df)

