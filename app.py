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
from sklearn.ensemble import GradientBoostingRegressor
import yfinance as yf

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB_PATH = "predictions.db"
FAVORITES_PATH = "favorites.json"
CONFIG_PATH = "config.json"

POSITIVE_KEYWORDS = ["上漲","漲停","突破","強勢","買進","獲利","成長","創高","利多","看好","增加","擴張","超越","創新高","上調","受惠","大漲","飆升","亮眼","優於預期"]
NEGATIVE_KEYWORDS = ["下跌","跌停","破底","弱勢","賣出","虧損","衰退","創低","利空","看壞","減少","縮減","低於預期","下調","警示","大跌","崩跌","不如預期","裁員","虧損"]


def load_config():
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return {"telegram": {"token": token, "chat_id": chat_id}}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# --- Database ---
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stop_loss (
            stock_id TEXT PRIMARY KEY,
            stock_name TEXT,
            stop_price REAL,
            target_price REAL,
            stop_triggered INTEGER DEFAULT 0,
            target_triggered INTEGER DEFAULT 0,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            stock_id TEXT PRIMARY KEY,
            sentiment_score REAL,
            headlines TEXT,
            label TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# --- Helpers ---
def load_favorites():
    with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]


def save_favorites(stocks):
    with open(FAVORITES_PATH, "w", encoding="utf-8") as f:
        json.dump({"stocks": stocks}, f, ensure_ascii=False, indent=2)


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


def predict_gbr(df, days=5):
    prices = df["close"].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    seq_len = min(30, len(scaled) - 1)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    last_seq = list(scaled[-seq_len:])
    predictions = []
    for _ in range(days):
        pred = model.predict([last_seq])[0]
        predictions.append(pred)
        last_seq = last_seq[1:] + [pred]
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()


# --- Real-time price ---
def get_realtime_price(stock_id: str):
    try:
        ticker = yf.Ticker(f"{stock_id}.TW")
        price = ticker.fast_info.last_price
        if price and price > 0:
            return round(float(price), 2)
    except:
        pass
    try:
        api = DataLoader()
        df = api.taiwan_stock_daily(stock_id=stock_id, start_date=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"))
        if not df.empty:
            return round(float(df["close"].iloc[-1]), 2)
    except:
        pass
    return None


# --- News sentiment ---
def analyze_sentiment(text: str) -> float:
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


def fetch_news_sentiment(stock_id: str, stock_name: str):
    conn = sqlite3.connect(DB_PATH)
    cached = conn.execute("SELECT sentiment_score, headlines, label, updated_at FROM news_cache WHERE stock_id=?", (stock_id,)).fetchone()
    if cached:
        updated = datetime.strptime(cached[3], "%Y-%m-%d %H:%M:%S")
        if datetime.now() - updated < timedelta(hours=6):
            conn.close()
            return {"score": cached[0], "headlines": json.loads(cached[1]), "label": cached[2]}
    conn.close()

    headlines = []
    scores = []
    try:
        url = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=50&page=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = http_requests.get(url, headers=headers, timeout=8)
        items = resp.json().get("data", {}).get("items", [])
        for item in items:
            title = item.get("title", "")
            if stock_name in title or stock_id in title:
                headlines.append(title)
                scores.append(analyze_sentiment(title))
    except:
        pass

    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    label = "正面" if avg_score > 0.1 else "負面" if avg_score < -0.1 else "中性"

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO news_cache (stock_id, sentiment_score, headlines, label, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (stock_id, avg_score, json.dumps(headlines[:5], ensure_ascii=False),
          label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return {"score": avg_score, "headlines": headlines[:5], "label": label}


# --- Stop-loss checker ---
def check_stop_loss():
    now = datetime.now()
    if now.weekday() >= 5:
        return
    if not (9 <= now.hour < 14):
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT stock_id, stock_name, stop_price, target_price, stop_triggered, target_triggered
        FROM stop_loss
        WHERE stop_price IS NOT NULL OR target_price IS NOT NULL
    """).fetchall()

    alerts = []
    for r in rows:
        stock_id, stock_name, stop_price, target_price, stop_triggered, target_triggered = r
        price = get_realtime_price(stock_id)
        if price is None:
            continue

        if stop_price and not stop_triggered and price <= stop_price:
            alerts.append(f"🚨 <b>停損警報</b>\n{stock_name}({stock_id})\n現價 <b>{price}</b> 已跌破停損價 {stop_price}")
            conn.execute("UPDATE stop_loss SET stop_triggered=1 WHERE stock_id=?", (stock_id,))

        if target_price and not target_triggered and price >= target_price:
            alerts.append(f"🎯 <b>目標達成</b>\n{stock_name}({stock_id})\n現價 <b>{price}</b> 已達目標價 {target_price}")
            conn.execute("UPDATE stop_loss SET target_triggered=1 WHERE stock_id=?", (stock_id,))

    conn.commit()
    conn.close()

    for alert in alerts:
        send_telegram(alert)


# --- Daily prediction ---
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
            preds = predict_gbr(df, days=1)
            predicted_price = round(preds[0], 2)
            direction = "up" if predicted_price > current_price else "down"

            conn.execute("""
                INSERT OR IGNORE INTO predictions
                (date, stock_id, stock_name, current_price, predicted_price, predicted_direction)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (today, stock["id"], stock["name"], current_price, predicted_price, direction))
            conn.commit()

            # 重置已觸發的停損/目標（每日重置）
            conn.execute("""
                UPDATE stop_loss SET stop_triggered=0, target_triggered=0
                WHERE stock_id=?
            """, (stock["id"],))
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
                        UPDATE predictions SET actual_price=?, actual_direction=?, is_correct=?
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
    http_requests.post(url, json={"chat_id": cfg["chat_id"], "text": message, "parse_mode": "HTML"})


def send_morning_report():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT p.stock_name, p.stock_id, p.current_price, p.predicted_price, p.predicted_direction,
               n.label, n.sentiment_score
        FROM predictions p
        LEFT JOIN news_cache n ON p.stock_id = n.stock_id
        WHERE p.date = ?
        ORDER BY p.stock_id
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
        sentiment = f" 【新聞:{r[5]}】" if r[5] else ""
        lines.append(f"  ↑ {r[0]}({r[1]})  {r[2]} → <b>{r[3]}</b>{sentiment}")

    lines.append("\n📉 <b>看跌</b>")
    for r in down_list:
        sentiment = f" 【新聞:{r[5]}】" if r[5] else ""
        lines.append(f"  ↓ {r[0]}({r[1]})  {r[2]} → <b>{r[3]}</b>{sentiment}")

    lines.append("\n─────────────────────")
    lines.append("🤖 AI 預測僅供參考，投資請謹慎")
    send_telegram("\n".join(lines))


# --- Scheduler ---
scheduler = BackgroundScheduler()
scheduler.add_job(run_daily_predictions, "cron", hour=18, minute=0)
scheduler.add_job(send_morning_report, "cron", hour=8, minute=0)
scheduler.add_job(check_stop_loss, "cron", minute="*/30")
scheduler.start()


# --- API Routes ---
class StockRequest(BaseModel):
    stock_id: str

class FavoriteRequest(BaseModel):
    id: str
    name: str

class StopLossRequest(BaseModel):
    stock_id: str
    stock_name: str
    stop_price: float = None
    target_price: float = None


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
        predictions = predict_gbr(df)
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


@app.get("/stock/lookup/{stock_id}")
async def lookup_stock(stock_id: str):
    try:
        api = DataLoader()
        df = api.taiwan_stock_info()
        row = df[df["stock_id"] == stock_id]
        if row.empty:
            return {"error": "找不到此股票代碼"}
        return {"id": stock_id, "name": str(row["stock_name"].iloc[0])}
    except Exception as e:
        return {"error": str(e)}


@app.post("/favorites/add")
async def add_favorite(req: FavoriteRequest):
    stocks = load_favorites()
    if any(s["id"] == req.id for s in stocks):
        return {"error": "已在清單中"}
    stocks.append({"id": req.id, "name": req.name})
    save_favorites(stocks)
    return {"status": "ok"}


@app.delete("/favorites/{stock_id}")
async def remove_favorite(stock_id: str):
    stocks = load_favorites()
    stocks = [s for s in stocks if s["id"] != stock_id]
    save_favorites(stocks)
    return {"status": "ok"}


@app.post("/favorites/predict-now")
async def predict_now():
    run_daily_predictions()
    return {"status": "ok", "message": "預測完成"}


@app.post("/notify/test")
async def notify_test():
    send_morning_report()
    return {"status": "ok", "message": "Telegram 已發送"}


# --- Stop-loss API ---
@app.get("/stop-loss")
async def get_stop_loss():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT stock_id, stock_name, stop_price, target_price, stop_triggered, target_triggered FROM stop_loss").fetchall()
    conn.close()
    return {r[0]: {"stock_name": r[1], "stop_price": r[2], "target_price": r[3],
                   "stop_triggered": bool(r[4]), "target_triggered": bool(r[5])} for r in rows}


@app.post("/stop-loss")
async def set_stop_loss(req: StopLossRequest):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO stop_loss (stock_id, stock_name, stop_price, target_price, stop_triggered, target_triggered, updated_at)
        VALUES (?, ?, ?, ?, 0, 0, ?)
    """, (req.stock_id, req.stock_name, req.stop_price, req.target_price, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/stop-loss/{stock_id}")
async def delete_stop_loss(stock_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM stop_loss WHERE stock_id=?", (stock_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/stop-loss/check-now")
async def check_now():
    check_stop_loss()
    return {"status": "ok"}


# --- News API ---
@app.get("/news/{stock_id}")
async def get_news(stock_id: str):
    stocks = load_favorites()
    stock = next((s for s in stocks if s["id"] == stock_id), None)
    name = stock["name"] if stock else stock_id
    result = fetch_news_sentiment(stock_id, name)
    return result


@app.post("/news/refresh-all")
async def refresh_all_news():
    favorites = load_favorites()
    for s in favorites:
        fetch_news_sentiment(s["id"], s["name"])
    return {"status": "ok"}


# --- Winrate & History ---
@app.get("/winrate")
async def get_winrate():
    conn = sqlite3.connect(DB_PATH)
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT stock_id, stock_name,
               COUNT(*) as total,
               SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) as correct,
               SUM(CASE WHEN is_correct IS NOT NULL THEN 1 ELSE 0 END) as judged
        FROM predictions WHERE date >= ?
        GROUP BY stock_id, stock_name ORDER BY stock_id
    """, (one_month_ago,)).fetchall()
    overall = conn.execute("""
        SELECT COUNT(*), SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN is_correct IS NOT NULL THEN 1 ELSE 0 END)
        FROM predictions WHERE date >= ?
    """, (one_month_ago,)).fetchone()
    conn.close()

    stats = [{"stock_id": r[0], "stock_name": r[1], "total": r[2], "correct": r[3], "judged": r[4],
              "winrate": round(r[3] / r[4] * 100, 1) if r[4] > 0 else None} for r in rows]
    total_judged = overall[2] or 0
    total_correct = overall[1] or 0
    return {"stats": stats, "overall": {
        "total": overall[0], "correct": total_correct, "judged": total_judged,
        "winrate": round(total_correct / total_judged * 100, 1) if total_judged > 0 else None
    }}


@app.get("/history")
async def get_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT date, stock_id, stock_name, current_price, predicted_price,
               predicted_direction, actual_price, actual_direction, is_correct
        FROM predictions ORDER BY date DESC, stock_id LIMIT 200
    """).fetchall()
    conn.close()
    return [{"date": r[0], "stock_id": r[1], "stock_name": r[2], "current_price": r[3],
             "predicted_price": r[4], "predicted_direction": r[5], "actual_price": r[6],
             "actual_direction": r[7], "is_correct": r[8]} for r in rows]
