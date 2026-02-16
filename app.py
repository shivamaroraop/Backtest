from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scipy.stats import linregress

# ----------------------------
# FastAPI Init
# ----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load Trained Model + Scaler
# ----------------------------

model = joblib.load("hmm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Feature Engineering
# ----------------------------

def calculate_rolling_slope(series, window=50):
    slopes = [np.nan] * window
    for i in range(window, len(series)):
        y = series[i-window:i].values
        x = np.arange(window)
        slope = linregress(x, y)[0]
        slopes.append(slope)
    return slopes


def prepare_features(ticker, years=5):
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return None

    data = df[['Close']].copy()
    data.columns = ['price']

    data['log_return'] = np.log(data['price'] / data['price'].shift(1))
    data['volatility_20'] = data['log_return'].rolling(20).std()
    data['trend_slope'] = calculate_rolling_slope(data['price'], 50)

    data.dropna(inplace=True)
    return data


# ----------------------------
# Backtest Logic
# ----------------------------

def backtest(data):
    features = data[['log_return', 'volatility_20', 'trend_slope']].values
    X_scaled = scaler.transform(features)

    states = model.predict(X_scaled)
    data['state'] = states

    # Label states by average return
    state_means = data.groupby('state')['log_return'].mean().sort_values()

    bear_state = state_means.index[0]
    sideways_state = state_means.index[1]
    bull_state = state_means.index[2]

    data['regime'] = data['state'].map({
        bull_state: "Bull",
        sideways_state: "Sideways",
        bear_state: "Bear"
    })

    # Strategy: Long only in Bull
    data['strategy_return'] = np.where(
        data['regime'] == "Bull",
        data['log_return'],
        0
    )

    data['buy_hold_curve'] = np.exp(data['log_return'].cumsum())
    data['strategy_curve'] = np.exp(data['strategy_return'].cumsum())

    return data


# ----------------------------
# API Endpoint
# ----------------------------

@app.get("/predict/{company}")
def predict_and_backtest(company: str):

    ticker = company.upper()
    data = prepare_features(ticker)

    if data is None:
        raise HTTPException(status_code=404, detail="Invalid ticker")

    data = backtest(data)

    buy_hold_return = (data['buy_hold_curve'].iloc[-1] - 1) * 100
    strategy_return = (data['strategy_curve'].iloc[-1] - 1) * 100

    latest_regime = data['regime'].iloc[-1]
    latest_price = float(data['price'].iloc[-1])

    return {
        "ticker": ticker,
        "latest_price": round(latest_price, 2),
        "latest_regime": latest_regime,
        "buy_and_hold_return_percent": round(buy_hold_return, 2),
        "model_strategy_return_percent": round(strategy_return, 2)
    }
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT"))
    uvicorn.run(app, host="0.0.0.0", port=port)
