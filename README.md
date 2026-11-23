# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program

# Step 1: Import the required libraries
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.stattools import adfuller

# Step 2: Set random seed and define periods
np.random.seed(42)

periods = 120

# Step 3: Create monthly time series
time = pd.date_range(
    start='2015-01-01',
    periods=periods,
    freq='ME'
)

# Step 4: Generate trend, seasonal and noise components
trend = np.linspace(10, 50, periods)

seasonal = 5 * np.sin(2 * np.pi * time.month / 12)

noise = np.random.normal(0, 2, periods)

# Step 5: Create the time series
series = pd.Series(
    trend + seasonal + noise,
    index=time
)

# Step 6: Difference once to remove trend
series_diff = series.diff().dropna()

# Step 7: Fit ARIMA model
model = ARIMA(series, order=(1, 1, 1))

model_fit = model.fit()

# Step 8: Display model summary
print(model_fit.summary())

# Step 9: Forecast the next 12 months
forecast = model_fit.get_forecast(steps=12)

# Step 10: Create forecast index
forecast_index = pd.date_range(
    start=series.index[-1] + pd.offsets.MonthEnd(1),
    periods=12,
    freq='ME'
)

forecast_mean = pd.Series(
    forecast.predicted_mean,
    index=forecast_index
)

# Step 11: Calculate confidence interval
conf_int = forecast.conf_int()

# Step 12: Plot the results
plt.figure(figsize=(10, 5))

plt.plot(series, label='Observed')
plt.plot(
    forecast_mean,
    label='Forecast',
    linestyle='--'
)

plt.fill_between(
    forecast_index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='gray',
    alpha=0.2
)

plt.legend()

plt.title('ARIMA(1,1,1) Forecast')

plt.xlabel('Year')

plt.ylabel('Values')

plt.show()


## Output

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/85f477fc-3755-48a8-bcc5-b7acea0f2a7f" />



## Result

Thus, the ARIMA (1,1,1) forecasting model was successfully implemented using Python. A synthetic time series dataset was created, and the model was fitted to historical data. The next 12 months were forecasted, and the observed values, predicted values, and confidence intervals were visualized using Matplotlib.
