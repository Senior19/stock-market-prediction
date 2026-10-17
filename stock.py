# Stock Market Prediction And Forecasting Using Stacked LSTM

# %%
### Keras and Tensorflow >2.0

# %%
### Data Collection
import os
from datetime import date

import numpy as np
import pandas as pd
import requests

key = os.getenv("TIINGO_API_KEY")
if not key:
    raise RuntimeError("Set the TIINGO_API_KEY environment variable before running this script.")

# %%
response = requests.get(
    "https://api.tiingo.com/tiingo/daily/AAPL/prices",
    headers={"Authorization": f"Token {key}"},
    params={"startDate": "2010-01-01", "endDate": date.today().isoformat()},
    timeout=30,
)
response.raise_for_status()
prices = response.json()
if not isinstance(prices, list) or not prices:
    raise ValueError("Tiingo returned no daily price data for AAPL.")
df = pd.DataFrame.from_records(prices)
if "close" not in df.columns:
    raise ValueError("Tiingo's response did not contain a 'close' price column.")

# %%
df.to_csv('AAPL.csv')

# %%
df1 = pd.to_numeric(df['close'], errors='coerce').dropna().to_numpy(dtype=float).reshape(-1, 1)
if not np.isfinite(df1).all():
    raise ValueError("The downloaded closing prices contain non-finite values.")

# %%
import matplotlib.pyplot as plt
plt.plot(df1)

# %%
### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 

# %%
from sklearn.preprocessing import MinMaxScaler
time_step = 100
training_size = int(len(df1) * 0.65)
if training_size <= time_step or training_size >= len(df1):
    raise ValueError(
        f"Need more than {time_step} training prices and at least one test price; "
        f"received {len(df1)} prices."
    )

scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(df1[:training_size])
scaled_prices = np.concatenate(
    (train_data, scaler.transform(df1[training_size:])),
    axis=0,
)

# %%
##splitting dataset into train and test split
test_data = scaled_prices[training_size - time_step:]

# %%
import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	if len(dataset) <= time_step:
		raise ValueError(f"Need more than {time_step} data points to create a dataset.")
	for i in range(len(dataset)-time_step):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# %%
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# %%
print(X_train.shape), print(y_train.shape)

# %%
print(X_test.shape), print(ytest.shape)

# %%
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

# %%
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

# %%
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

# %%
model.summary()

# %%
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-6,
    restore_best_weights=True,
)
model.fit(
    X_train,
    y_train,
    validation_data=(X_test,ytest),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1,
)

# %%
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

# %%
##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

# %%
### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
math.sqrt(mean_squared_error(y_train_actual, train_predict))

# %%
### Test Data RMSE
ytest_actual = scaler.inverse_transform(ytest.reshape(-1, 1))
math.sqrt(mean_squared_error(ytest_actual, test_predict))

# %%
### Plotting 
# shift train predictions for plotting
look_back = time_step
trainPredictPlot = numpy.empty_like(scaled_prices)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(scaled_prices)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[training_size:, :] = test_predict
# plot baseline and predictions
plt.plot(df1, label="Actual closing price")
plt.plot(trainPredictPlot, label="Training predictions")
plt.plot(testPredictPlot, label="Test predictions")
plt.axvline(training_size, color="gray", linestyle="--", label="Train/test split")
plt.xlabel("Trading day")
plt.ylabel("Closing price")
plt.title("AAPL actual prices and model predictions")
plt.legend()
plt.show()

# %%
n_steps = time_step
x_input = scaled_prices[-n_steps:].reshape(1, n_steps, 1)

# %%
# demonstrate prediction for the next 30 days

lst_output = []
forecast_horizon = 30
for _ in range(forecast_horizon):
    next_value = model.predict(x_input, verbose=0)
    lst_output.append(next_value[0, 0])
    x_input = np.concatenate((x_input[:, 1:, :], next_value.reshape(1, 1, 1)), axis=1)

# %%
day_new=np.arange(1, n_steps + 1)
day_pred=np.arange(n_steps + 1, n_steps + forecast_horizon + 1)

# %%
plt.plot(day_new, df1[-n_steps:], label="Recent actual prices")
forecast = scaler.inverse_transform(np.asarray(lst_output).reshape(-1, 1))
plt.plot(day_pred, forecast, label="30-day forecast")
plt.axvline(n_steps + 0.5, color="gray", linestyle="--", label="Forecast starts")
plt.xlabel("Trading day (relative)")
plt.ylabel("Closing price")
plt.title("Recent AAPL prices and 30-day forecast")
plt.legend()
plt.show()

# %%
df3 = np.concatenate((df1, forecast), axis=0)
plt.plot(np.arange(len(df1)), df1[:, 0], label="Historical closing price")
plt.plot(
    np.arange(len(df1) - 1, len(df3)),
    df3[len(df1) - 1 :, 0],
    label="Forecast",
)
plt.axvline(len(df1) - 0.5, color="gray", linestyle="--", label="Forecast starts")
plt.xlabel("Trading day")
plt.ylabel("Closing price")
plt.title("AAPL history with forecast")
plt.legend()
plt.show()
