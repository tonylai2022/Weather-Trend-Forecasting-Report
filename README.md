# **Weather Trend Forecasting Report**

This repository contains scripts for Exploratory Data Analysis (EDA), model development, and evaluation to build a model for predicting temperature trends. The EDA script analyzes global weather data, focusing on temperature trends and related features, and generates visualizations such as global temperature trends, histograms, correlation matrices, and time series plots. The model development and evaluation scripts aim to create and assess a predictive model for temperature forecasting.

## Prerequisites

- **Python 3.x**: Ensure you have Python installed (recommended version 3.8 or higher).
- **Required Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `statsmodels`

## Setup and Installation  

### Step 1: Clone the Repository 
1. Open your terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Clone the repository using the following command:
    ```bash
    git clone https://github.com/tonylai2022/Weather-Trend-Forecasting-Report.git
    ```

### Step 2: Navigate to the Repository
```bash
cd Weather-Trend-Forecasting-Report
```

### Step 3: Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Code

### Exploratory Data Analysis (EDA)
```bash
python eda.py
```

### Model Development and Evaluation
```bash
python model.py
```

## Methodology

In this project, three models were built to predict temperature trends:
1. **Ordinary Least Squares (OLS)**
2. **AutoRegressive Integrated Moving Average (ARIMA)**
3. **Holt-Winters Exponential Smoothing**

The best model was selected by comparing the Root Mean Squared Error (RMSE) and backtest RMSE of each model.

## Results

The Ordinary Least Squares (OLS) model was found to be the best model based on the RMSE and backtest RMSE comparisons. For details, please read the Weather Trend Forecasting Report in the repo.