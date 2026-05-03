from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
import sqlite3
import json
import os
import requests as http_requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from FinMind.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB_PATH = "predictions.db"
FAVORITES_PATH = "favorites.json"
CONFIG_PATH = "config.json"


def load_config():
    # 優先使用環境變數（部署用），其次讀本機 config.json
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return {"telegram": {"token": token, "chat_id": chat_id}}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# --- Database setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            stock_name TEXT NOT NULL,
            current_price REAL,
            predicted_price REAL,
            predicted_direction TEXT,
            actual_price REAL,
            actual_direction TEXT,
            is_correct INTEGER,
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            UNIQUE(date, stock_id)
        )
    """)
    conn.commit()
    conn.close()

init_db()


# --- Helpers ---
def load_favorites():
    with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]


def fetch_stock_data(stock_id: str):
    api = DataLoader()
    start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start)
    return df.sort_values("date").reset_index(drop=True)


def calc_indicators(df):
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df


def predict_lstm(df, days=5):
    prices = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    seq_len = min(30, len(scaled) - 1)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    last_seq = scaled[-seq_len:]
    predictions = []
    for _ in range(days):
        inp = last_seq.reshape(1, seq_len, 1)
        pred = model.predict(inp, verbose=0)[0][0]
        predictions.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()


# --- Daily prediction job ---
def run_daily_predictions():
    favorites = load_favorites()
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)

    for stock in favorites:
        try:
            df = fetch_stock_data(stock["id"])
            if df.empty or len(df) < 35:
                continue
            current_price = float(df["close"].iloc[-1])
            preds = predict_lstm(df, days=1)
            predicted_price = round(preds[0], 2)
            direction = "up" if predicted_price > current_price else "down"

            conn.execute("""
                INSERT OR IGNORE INTO predictions
                (date, stock_id, stock_name, current_price, predicted_price, predicted_direction)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (today, stock["id"], stock["name"], current_price, predicted_price, direction))
            conn.commit()
            print(f"[{today}] {stock['name']} 預測完成: {current_price} → {predicted_price} ({direction})")
        except Exception as e:
            print(f"[{today}] {stock['name']} 錯誤: {e}")

    conn.close()
    update_actual_prices()


def update_actual_prices():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, date, stock_id, predicted_direction
        FROM predictions
        WHERE actual_price IS NULL AND date < date('now', 'localtime')
    """).fetchall()

    for row in rows:
        pred_id, pred_date, stock_id, pred_dir = row
        try:
            api = DataLoader()
            df = api.taiwan_stock_daily(stock_id=stock_id, start_date=pred_date)
            df = df.sort_values("date")
            dates = df["date"].astype(str).tolist()
            if pred_date in dates:
                idx = dates.index(pred_date)
                if idx > 0:
                    prev_close = float(df["close"].iloc[idx - 1])
                    actual_close = float(df["close"].iloc[idx])
                    actual_dir = "up" if actual_close > prev_close else "down"
                    is_correct = 1 if actual_dir == pred_dir else 0
                    conn.execute("""
                        UPDATE predictions
                        SET actual_price=?, actual_direction=?, is_correct=?
                        WHERE id=?
                    """, (actual_close, actual_dir, is_correct, pred_id))
            conn.commit()
        except Exception as e:
            print(f"更新實際價格失敗 {stock_id}: {e}")

    conn.close()


# --- Telegram ---
def send_telegram(message: str):
    cfg = load_config()["telegram"]
    url = f"https://api.telegram.org/bot{cfg['token']}/sendMessage"
    http_requests.post(url, json={
        "chat_id": cfg["chat_id"],
        "text": message,
        "parse_mode": "HTML"
    })


def send_morning_report():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT stock_name, stock_id, current_price, predicted_price, predicted_direction
        FROM predictions
        WHERE date = ?
        ORDER BY stock_id
    """, (today,)).fetchall()
    conn.close()

    if not rows:
        send_telegram(f"📊 <b>{today} 股票預測</b>\n\n今日尚無預測資料，請先執行預測。")
        return

    up_list = [r for r in rows if r[4] == "up"]
    down_list = [r for r in rows if r[4] == "down"]

    lines = [f"📊 <b>{today} 每日股票預測報告</b>\n"]
    lines.append(f"看漲 {len(up_list)} 支 ｜ 看跌 {len(down_list)} 支\n")
    lines.append("─────────────────────")

    lines.append("\n📈 <b>看漲</b>")
    for r in up_list:
        lines.append(f"  ↑ {r[0]}({r[1]})  {r[2]} → <b>{r[3]}</b>")

    lines.append("\n📉 <b>看跌</b>")
    for r in down_list:
        lines.append(f"  ↓ {r[0]}({r[1]})  {r[2]} → <b>{r[3]}</b>")

    lines.append("\n─────────────────────")
    lines.append("🤖 AI 預測僅供參考，投資請謹慎")

    send_telegram("\n".join(lines))


# --- Scheduler ---
scheduler = BackgroundScheduler()
scheduler.add_job(run_daily_predictions, "cron", hour=18, minute=0)
scheduler.add_job(send_morning_report, "cron", hour=8, minute=0)
scheduler.start()


# --- API Routes ---
class StockRequest(BaseModel):
    stock_id: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(req: StockRequest):
    try:
        df = fetch_stock_data(req.stock_id)
        if df.empty:
            return {"error": "找不到此股票代碼"}
        df = calc_indicators(df)
        recent = df.tail(60).copy()
        predictions = predict_lstm(df)
        last_date = pd.to_datetime(df["date"].iloc[-1])
        future_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(5)]
        close = recent["close"].tolist()
        return {
            "stock_id": req.stock_id,
            "dates": recent["date"].astype(str).tolist(),
            "close": close,
            "ma5": recent["MA5"].tolist(),
            "ma20": recent["MA20"].tolist(),
            "rsi": recent["RSI"].tolist(),
            "macd": recent["MACD"].tolist(),
            "signal": recent["Signal"].tolist(),
            "pred_dates": future_dates,
            "predictions": predictions,
            "current_price": close[-1],
            "pred_price": round(predictions[0], 2),
            "trend": "↑ 看漲" if predictions[0] > close[-1] else "↓ 看跌"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/favorites")
async def get_favorites():
    return load_favorites()


@app.post("/favorites/predict-now")
async def predict_now():
    run_daily_predictions()
    return {"status": "ok", "message": "預測完成"}


@app.post("/notify/test")
async def notify_test():
    send_morning_report()
    return {"status": "ok", "message": "Telegram 已發送"}


@app.get("/winrate")
async def get_winrate():
    conn = sqlite3.connect(DB_PATH)
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    rows = conn.execute("""
        SELECT stock_id, stock_name,
               COUNT(*) as total,
               SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) as correct,
               SUM(CASE WHEN is_correct IS NOT NULL THEN 1 ELSE 0 END) as judged
        FROM predictions
        WHERE date >= ?
        GROUP BY stock_id, stock_name
        ORDER BY stock_id
    """, (one_month_ago,)).fetchall()

    overall = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) as correct,
               SUM(CASE WHEN is_correct IS NOT NULL THEN 1 ELSE 0 END) as judged
        FROM predictions
        WHERE date >= ?
    """, (one_month_ago,)).fetchone()

    conn.close()

    stats = []
    for r in rows:
        winrate = round(r[3] / r[4] * 100, 1) if r[4] > 0 else None
        stats.append({
            "stock_id": r[0], "stock_name": r[1],
            "total": r[2], "correct": r[3], "judged": r[4],
            "winrate": winrate
        })

    total_judged = overall[2] or 0
    total_correct = overall[1] or 0
    overall_winrate = round(total_correct / total_judged * 100, 1) if total_judged > 0 else None

    return {
        "stats": stats,
        "overall": {
            "total": overall[0], "correct": total_correct,
            "judged": total_judged, "winrate": overall_winrate
        }
    }


@app.get("/history")
async def get_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT date, stock_id, stock_name, current_price, predicted_price,
               predicted_direction, actual_price, actual_direction, is_correct
        FROM predictions
        ORDER BY date DESC, stock_id
        LIMIT 200
    """).fetchall()
    conn.close()
    return [
        {
            "date": r[0], "stock_id": r[1], "stock_name": r[2],
            "current_price": r[3], "predicted_price": r[4],
            "predicted_direction": r[5], "actual_price": r[6],
            "actual_direction": r[7], "is_correct": r[8]
        }
        for r in rows
    ]
