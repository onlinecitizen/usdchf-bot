import yfinance as yf
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from datetime import datetime

# --- 1. TELEGRAM DETAILS ---
TOKEN = os.getenv("TOKEN")  # safer: set in Render environment variables
CHAT_ID = os.getenv("CHAT_ID")
MODEL_FILE = "xgboost_model.json"
LOG_FILE = "usdchf_signals.csv"

def send_to_phone(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def log_signal(signal_type, confidence, entry, stop_loss, take_profit):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal": signal_type,
        "confidence": confidence,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }
    df = pd.DataFrame([record])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# --- 2. GET DATA ---
data = yf.download("USDCHF=X", period="5y", interval="1d", auto_adjust=True, progress=False)
df = data.copy()

# --- 3. FEATURE ENGINEERING ---
df['return'] = df['Close'].pct_change()
df['volatility'] = df['return'].rolling(20).std()
df['momentum'] = df['return'].rolling(10).mean()
df['trend_strength'] = df['return'].rolling(20).mean() / df['volatility']
df['lag1'] = df['return'].shift(1)
df['lag5'] = df['return'].shift(5)
df['lag20'] = df['return'].shift(20)
df['regime'] = (df['volatility'] > df['volatility'].median()).astype(int)

# ATR
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift())
low_close = np.abs(df['Low'] - df['Close'].shift())
df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

# Target
df['target'] = (df['return'].shift(-1) > 0).astype(int)
df = df.dropna()

features = ['volatility','momentum','trend_strength','lag1','lag5','lag20','regime']
X = df[features]
y = df['target']

# --- 4. TRAIN & SAVE MODEL (first run only) ---
if not os.path.exists(MODEL_FILE):
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    model.save_model(MODEL_FILE)
    print("✅ Model trained and saved")

# --- 5. LOAD MODEL ---
final_model = xgb.XGBClassifier()
final_model.load_model(MODEL_FILE)

# --- 6. PREDICTION ---
latest = df.iloc[[-1]]
current_p = latest['Close'].values[0]
atr_v = latest['ATR'].values[0]
prob_up = final_model.predict_proba(latest[features])[:, 1][0]

sl_dist = atr_v * 1.5
tp_dist = atr_v * 3.0

if prob_up > 0.62:
    msg = (f"🚀 *USDCHF BUY SIGNAL*\n\n"
           f"📈 *Confidence:* {prob_up:.1%}\n"
           f"💵 *Entry:* {current_p:.5f}\n"
           f"🛑 *Stop Loss:* {current_p - sl_dist:.5f}\n"
           f"🎯 *Take Profit:* {current_p + tp_dist:.5f}")
    log_signal("BUY", prob_up, current_p, current_p - sl_dist, current_p + tp_dist)
elif prob_up < 0.38:
    msg = (f"📉 *USDCHF SELL SIGNAL*\n\n"
           f"📊 *Confidence:* {(1-prob_up):.1%}\n"
           f"💵 *Entry:* {current_p:.5f}\n"
           f"🛑 *Stop Loss:* {current_p + sl_dist:.5f}\n"
           f"🎯 *Take Profit:* {current_p - tp_dist:.5f}")
    log_signal("SELL", 1-prob_up, current_p, current_p + sl_dist, current_p - tp_dist)
else:
    msg = (f"😴 *USDCHF: NO TRADE*\n\n"
           f"Neutral market (Prob: {prob_up:.1%}).\n"
           f"Current Price: {current_p:.5f}")
    log_signal("NO TRADE", prob_up, current_p, None, None)

send_to_phone(msg)
print("✅ Signal sent to Telegram and logged")
