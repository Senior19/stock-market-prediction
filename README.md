# Stock Price Prediction

This project downloads Apple's daily closing prices from Tiingo, trains a stacked long short-term memory (LSTM) neural network, and plots historical prices alongside model predictions and a 30-step forecast.

> **Disclaimer:** This is an educational example, not financial advice. Forecasts from a model trained on past prices are uncertain and should not be used on their own to make investment decisions.

## What the script does

[`stock.py`](./stock.py) runs the following workflow:

1. Downloads AAPL daily price data from Tiingo, starting on 2010-01-01 and ending on the current date.
2. Saves the returned data to `AAPL.csv` in the project directory.
3. Uses the `close` prices, scales them to the 0–1 range, and splits the time series into 65% training data and 35% test data.
4. Builds input sequences from the previous 100 observations and trains a three-layer stacked LSTM. Training uses early stopping based on validation loss.
5. Makes training and test predictions, calculates RMSE in the corresponding code cells, and displays a comparison plot.
6. Iteratively predicts the next 30 observations and displays forecast plots.

The forecast horizon is **30 observations (trading days in the source data)**, not 30 calendar days. The script currently targets AAPL directly; changing the ticker requires changing the Tiingo request and plot labels.

## Requirements

- Python with versions compatible with the packages in [`requirements.txt`](./requirements.txt)
- A Tiingo account and API token with access to daily stock-price data
- Internet access to reach the Tiingo API

The project dependencies are listed in `requirements.txt`: Matplotlib, NumPy, pandas, Requests, scikit-learn, and TensorFlow.

## Setup

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set your Tiingo API token in the environment. In PowerShell, replace the placeholder with your token:

```powershell
$env:TIINGO_API_KEY = "your-tiingo-api-token"
```

The script reads this environment variable at startup. Keep the token private and do not commit it to source control.

## Run

With the virtual environment active and `TIINGO_API_KEY` set:

```powershell
python stock.py
```

The script prints dataset shapes and the model summary, shows training progress, and opens Matplotlib plots. It also writes or overwrites `AAPL.csv` in the current working directory. Run it in an environment that supports interactive Matplotlib windows.

`stock.py` is organized into `# %%` code cells. When running individual cells in an interactive Python editor or notebook-style workflow, the RMSE expressions are displayed as cell results. When running the file as a regular Python script, those expressions are evaluated but are not explicitly printed.

## Notes and limitations

- The model uses only the closing-price series; it does not include volume, company fundamentals, or market-wide features.
- The test segment is also used as validation data for early stopping.
- The 30-step forecast feeds each prediction back into the model as the next input, so errors can compound over the forecast horizon.
- The forecast is generated in scaled-price space and converted back to the original price scale for plotting.
- The script requires enough returned observations to create 100-step input windows in both the training and test portions. A short or empty API response will stop execution with an error.
