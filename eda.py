import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
df = pd.read_csv("GlobalWeatherRepository.csv")

# Convert 'last_updated' to datetime and sort
df['last_updated'] = pd.to_datetime(df['last_updated'])
df = df.sort_values('last_updated')

# Add lagged temperature (temp_t-1) by location
df['temp_t-1'] = df.groupby('location_name')['temperature_celsius'].shift(1)
df.dropna(subset=['temp_t-1', 'uv_index', 'humidity', 'temperature_celsius'], inplace=True)

# EDA: Basic Statistics
print("Basic Statistics of the Dataset:")
print(df[['temperature_celsius', 'temp_t-1', 'uv_index', 'humidity']].describe())
print("\nDataset Info:")
print(df.info())

# EDA: Global Temperature Trend Over Time
global_avg = df.groupby('last_updated')['temperature_celsius'].mean().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(global_avg['last_updated'], global_avg['temperature_celsius'], label='Global Avg Temperature (°C)', color='orange')
plt.title('Global Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.savefig('figure_6_global_temperature_trend.png')
plt.show()

# EDA: Correlation Analysis
numerical_cols = ['temperature_celsius', 'temp_t-1', 'uv_index', 'humidity']
corr_matrix = df[numerical_cols].corr()
temp_correlations = corr_matrix['temperature_celsius'].drop('temperature_celsius')
print("\n=== Correlations with Temperature (°C) ===")
print(temp_correlations)

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Selected Features')
plt.savefig('figure_1_correlation_heatmap.png')
plt.show()



#Three Histograms in One Row (temp_t-1, uv_index, humidity)
plt.figure(figsize=(15, 4))

# Histogram for temp_t-1
plt.subplot(1, 3, 1)
sns.histplot(data=df, x='temp_t-1', bins=30, kde=True, color='blue')
plt.title('Histogram of Temp_t-1 (°C)')
plt.xlabel('Temp_t-1 (°C)')
plt.ylabel('Count')

# Histogram for uv_index
plt.subplot(1, 3, 2)
sns.histplot(data=df, x='uv_index', bins=30, kde=True, color='blue')
plt.title('Histogram of UV Index')
plt.xlabel('UV Index')
plt.ylabel('Count')

# Histogram for humidity
plt.subplot(1, 3, 3)
sns.histplot(data=df, x='humidity', bins=30, kde=True, color='blue')
plt.title('Histogram of Humidity (%)')
plt.xlabel('Humidity (%)')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('figure_2f_three_histograms_in_row.png')
plt.show()

# NEW: Time Series Trends and Cycles (ACF and PACF Plots)
global_avg = df.groupby('last_updated')['temperature_celsius'].mean()

plt.figure(figsize=(12, 5))

# ACF Plot
plt.subplot(1, 2, 1)
plot_acf(global_avg, lags=20, ax=plt.gca(), alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Temperature (°C)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

# PACF Plot
plt.subplot(1, 2, 2)
plot_pacf(global_avg, lags=20, ax=plt.gca(), alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Temperature (°C)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')

plt.suptitle('Time Series Trends and Cycles')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figure_3_acf_pacf_plots.png')
plt.show()

