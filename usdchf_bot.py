# --- 6. PREDICTION & SIGNAL GENERATION ---
# Use .item() or [0] to ensure these are single floats, not arrays
latest = df.iloc[-1:] 
current_p = float(latest['Close'].iloc[0])
atr_v = float(latest['ATR'].iloc[0])

# Get probability as a single float
prob_up_raw = final_model.predict_proba(latest[features])[:, 1][0]
prob_up = float(prob_up_raw)

# Risk Calculations
sl_dist = atr_v * 1.5
tp_dist = atr_v * 3.0

pair_name = "USD/CHF"

if prob_up > 0.62:
    msg = (f"🚀 *{pair_name} BUY SIGNAL*\n\n"
           f"📈 *Confidence:* {prob_up:.1%}\n"
           f"💵 *Entry:* {current_p:.5f}\n"
           f"🛑 *Stop Loss:* {current_p - sl_dist:.5f}\n"
           f"🎯 *Take Profit:* {current_p + tp_dist:.5f}")
    log_signal("BUY", prob_up, current_p, current_p - sl_dist, current_p + tp_dist)

elif prob_up < 0.38:
    confidence = 1 - prob_up
    msg = (f"📉 *{pair_name} SELL SIGNAL*\n\n"
           f"📊 *Confidence:* {confidence:.1%}\n"
           f"💵 *Entry:* {current_p:.5f}\n"
           f"🛑 *Stop Loss:* {current_p + sl_dist:.5f}\n"
           f"🎯 *Take Profit:* {current_p - tp_dist:.5f}")
    log_signal("SELL", confidence, current_p, current_p + sl_dist, current_p - tp_dist)

else:
    msg = (f"😴 *{pair_name}: NO TRADE*\n\n"
           f"Neutral market (Prob: {prob_up:.1%}).\n"
           f"Current Price: {current_p:.5f}")
    log_signal("NO TRADE", prob_up, current_p, None, None)

send_to_phone(msg)
print(f"✅ {pair_name} Signal sent to Telegram and logged")
