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
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import yfinance as yf
import threading
import anthropic

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
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
        conn.commit()
    except Exception:
        pass  # column already exists
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
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start)
    return df.sort_values("date").reset_index(drop=True)


def fetch_institutional(stock_id: str):
    try:
        api = DataLoader()
        start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        df["net"] = df["buy"] - df["sell"]
        pivot = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
        pivot.columns = [f"inst_{c}" for c in pivot.columns]
        pivot = pivot.reset_index()
        return pivot
    except:
        return pd.DataFrame()


def fetch_taiex():
    try:
        api = DataLoader()
        start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        df = api.taiwan_stock_daily(stock_id="IR0001", start_date=start)
        df = df[["date", "close"]].rename(columns={"close": "taiex"})
        return df
    except:
        return pd.DataFrame()


def fetch_margin_data(stock_id: str):
    try:
        api = DataLoader()
        start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start)
        if df is None or df.empty:
            return None
        df = df.copy()
        buy_col = next((c for c in df.columns if "Buy" in c and "Margin" in c), None)
        sell_col = next((c for c in df.columns if "Sell" in c and "Short" in c), None)
        if buy_col and sell_col:
            df["margin_net"] = df[buy_col].fillna(0) - df[sell_col].fillna(0)
        else:
            df["margin_net"] = 0.0
        return df[["date", "margin_net"]].sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def calc_indicators(df):
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df


def build_features(df, inst_df=None, taiex_df=None, sentiment_score=0.0, margin_df=None):
    d = df.copy()

    # 技術指標
    d["MA5"] = d["close"].rolling(5).mean()
    d["MA20"] = d["close"].rolling(20).mean()
    d["MA60"] = d["close"].rolling(60).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + gain / loss))
    ema12 = d["close"].ewm(span=12).mean()
    ema26 = d["close"].ewm(span=26).mean()
    d["MACD"] = ema12 - ema26
    d["Signal"] = d["MACD"].ewm(span=9).mean()

    # 布林通道
    d["BB_mid"] = d["close"].rolling(20).mean()
    d["BB_std"] = d["close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * d["BB_std"]
    d["BB_lower"] = d["BB_mid"] - 2 * d["BB_std"]
    d["BB_pct"] = (d["close"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"] + 1e-9)

    # 價格動能
    d["ret1"] = d["close"].pct_change(1)
    d["ret5"] = d["close"].pct_change(5)
    d["ret20"] = d["close"].pct_change(20)

    # 成交量特徵
    d["vol_ma5"] = d["Trading_Volume"].rolling(5).mean()
    d["vol_ratio"] = d["Trading_Volume"] / (d["vol_ma5"] + 1)

    # MA 交叉
    d["ma5_20"] = d["MA5"] / (d["MA20"] + 1e-9)
    d["price_ma20"] = d["close"] / (d["MA20"] + 1e-9)

    # 星期幾
    d["weekday"] = pd.to_datetime(d["date"]).dt.weekday

    # 新聞情緒
    d["sentiment"] = sentiment_score

    # 大盤
    if taiex_df is not None and not taiex_df.empty:
        d = d.merge(taiex_df, on="date", how="left")
        d["taiex"] = d["taiex"].ffill()
        d["taiex_ret"] = d["taiex"].pct_change(1)
    else:
        d["taiex_ret"] = 0.0

    # 法人籌碼
    if inst_df is not None and not inst_df.empty:
        d = d.merge(inst_df, on="date", how="left").fillna(0)

    # 融資融券淨額
    if margin_df is not None and not margin_df.empty:
        d = d.merge(margin_df, on="date", how="left")
        d["margin_net"] = d["margin_net"].fillna(0)
    else:
        d["margin_net"] = 0.0

    # 法人思維特徵
    foreign_col = next((c for c in d.columns if "外資" in c or "Foreign" in c), None)
    trust_col   = next((c for c in d.columns if "投信" in c or "Investment_Trust" in c or ("Trust" in c and "inst_" in c)), None)
    dealer_col  = next((c for c in d.columns if "自營" in c or "Dealer" in c), None)

    if foreign_col:
        d["foreign_net_5d"]  = d[foreign_col].rolling(5).sum()
        d["foreign_net_20d"] = d[foreign_col].rolling(20).sum()
        is_buy = d[foreign_col] > 0
        grp = (is_buy != is_buy.shift()).cumsum()
        streak = is_buy.groupby(grp).cumcount() + 1
        d["foreign_consecutive"] = streak * np.where(is_buy, 1, -1)
    else:
        d["foreign_net_5d"] = d["foreign_net_20d"] = d["foreign_consecutive"] = 0.0

    d["trust_net_5d"]  = d[trust_col].rolling(5).sum()  if trust_col  else 0.0
    d["dealer_net_5d"] = d[dealer_col].rolling(5).sum() if dealer_col else 0.0

    if foreign_col and trust_col and dealer_col:
        total = d[foreign_col] + d[trust_col] + d[dealer_col]
        d["inst_agree"]    = ((np.sign(d[foreign_col]) == np.sign(d[trust_col])) &
                              (np.sign(d[trust_col])   == np.sign(d[dealer_col]))).astype(float)
        d["inst_total_5d"] = total.rolling(5).sum()
        if "Trading_Volume" in d.columns:
            d["inst_vol_ratio"] = total / (d["Trading_Volume"].abs() + 1)
        else:
            d["inst_vol_ratio"] = 0.0
    else:
        d["inst_agree"] = d["inst_total_5d"] = d["inst_vol_ratio"] = 0.0

    return d


def predict_xgb(df, inst_df=None, taiex_df=None, sentiment_score=0.0, days=5):
    d = build_features(df, inst_df, taiex_df, sentiment_score)

    feature_cols = ["MA5", "MA20", "MA60", "RSI", "MACD", "Signal",
                    "BB_pct", "ret1", "ret5", "ret20",
                    "vol_ratio", "ma5_20", "price_ma20",
                    "weekday", "sentiment", "taiex_ret"]
    inst_cols = [c for c in d.columns if c.startswith("inst_")]
    feature_cols += inst_cols

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=feature_cols + ["close"])
    if len(d) < 40:
        return [float(df["close"].iloc[-1])] * days

    X = d[feature_cols].values
    y = d["close"].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, random_state=42
    )
    model.fit(X_scaled, y_scaled)

    # 預測未來 N 天
    last_row = d[feature_cols].iloc[-1].values.copy()
    predictions = []
    for _ in range(days):
        x_in = scaler_X.transform([last_row])
        pred_scaled = model.predict(x_in)[0]
        pred_price = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(round(float(pred_price), 2))
        last_row[feature_cols.index("ret1")] = (pred_price - float(d["close"].iloc[-1])) / float(d["close"].iloc[-1])

    return predictions


def get_stock_winrate_map():
    conn = sqlite3.connect(DB_PATH)
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT stock_id,
               SUM(CASE WHEN is_correct=1 THEN 1.0 ELSE 0 END) /
               NULLIF(SUM(CASE WHEN is_correct IS NOT NULL THEN 1 ELSE 0 END), 0)
        FROM predictions WHERE date >= ? GROUP BY stock_id
    """, (one_month_ago,)).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows if r[1] is not None}


def get_taiex_daily_change():
    try:
        df = fetch_taiex()
        if df is not None and len(df) >= 2:
            c = df["close"].values
            return (float(c[-1]) - float(c[-2])) / float(c[-2])
    except Exception:
        pass
    return 0.0


# --- C1: 信心度動態倉位建議 ---
def calc_position_size(confidence: float, direction: str,
                       winrate: float = None, taiex_change: float = 0.0) -> dict:
    if direction == "neutral" or confidence < 0.55:
        return {"ratio": 0.0, "label": "不建議進場", "reason": "方向不明或信心不足"}

    if confidence >= 0.80:
        ratio = 1.0
    elif confidence >= 0.65:
        ratio = 0.5
    else:
        ratio = 0.25

    reasons = []
    if winrate is not None and winrate < 0.40:
        ratio = max(0.0, ratio - 0.25)
        reasons.append(f"勝率偏低({winrate*100:.0f}%)降倉")
    if taiex_change < -0.02:
        ratio = max(0.0, ratio - 0.25)
        reasons.append("大盤弱勢降倉")

    if ratio <= 0:
        return {"ratio": 0.0, "label": "不建議進場", "reason": "、".join(reasons)}

    if ratio >= 1.0:
        label = "全倉"
    elif ratio >= 0.5:
        label = "半倉"
    else:
        label = "輕倉"

    return {"ratio": ratio, "label": label, "reason": "、".join(reasons)}


def predict_with_confidence(df, inst_df=None, taiex_df=None, margin_df=None,
                             sentiment_score=0.0, stock_winrate=None):
    d = build_features(df, inst_df, taiex_df, sentiment_score, margin_df)

    base_cols = ["MA5", "MA20", "MA60", "RSI", "MACD", "Signal",
                 "BB_pct", "ret1", "ret5", "ret20",
                 "vol_ratio", "ma5_20", "price_ma20",
                 "weekday", "sentiment", "taiex_ret", "margin_net",
                 "foreign_net_5d", "foreign_net_20d", "foreign_consecutive",
                 "trust_net_5d", "dealer_net_5d",
                 "inst_agree", "inst_total_5d", "inst_vol_ratio"]
    inst_cols = [c for c in d.columns if c.startswith("inst_")]
    feature_cols = base_cols + [c for c in inst_cols if c not in base_cols]

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=feature_cols + ["close"])

    N = len(d)
    horizon = 1
    if N < 40 + horizon:
        price = float(df["close"].iloc[-1])
        return {"price": price, "direction": "up", "confidence": 0.5}

    X_all = d[feature_cols].values
    closes = d["close"].values
    X_train = X_all[:N - horizon]
    y_reg = closes[:N - horizon]
    y_cls = (closes[horizon:N] > closes[:N - horizon]).astype(int)

    # 根據勝率調整模型複雜度（#1 自適應調參）
    if stock_winrate is not None and stock_winrate < 0.40:
        params = dict(max_iter=200, max_depth=3, learning_rate=0.03, min_samples_leaf=8, random_state=42)
    elif stock_winrate is not None and stock_winrate > 0.60:
        params = dict(max_iter=400, max_depth=5, learning_rate=0.05, min_samples_leaf=4, random_state=42)
    else:
        params = dict(max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=5, random_state=42)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    y_reg_scaled = scaler_y.fit_transform(y_reg.reshape(-1, 1)).flatten()

    reg = HistGradientBoostingRegressor(**params)
    reg.fit(X_scaled, y_reg_scaled)

    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X_scaled, y_cls)

    X_last = scaler_X.transform(X_all[-1:])
    pred_price = float(scaler_y.inverse_transform(reg.predict(X_last).reshape(-1, 1))[0][0])
    proba = clf.predict_proba(X_last)[0]
    raw_confidence = float(max(proba))
    direction = "up" if proba[1] > proba[0] else "down"

    # A1: Walk-forward 時序驗證，校準信心度
    wf_accuracy = 0.5
    val_size = max(10, N // 5)
    train_end = N - horizon - val_size
    if train_end >= 40:
        try:
            X_wf = X_all[:train_end]
            y_wf_cls = (closes[horizon:train_end + horizon] > closes[:train_end]).astype(int)
            scaler_wf = MinMaxScaler()
            X_wf_scaled = scaler_wf.fit_transform(X_wf)
            clf_wf = HistGradientBoostingClassifier(**params)
            clf_wf.fit(X_wf_scaled, y_wf_cls)
            X_val = scaler_wf.transform(X_all[train_end:N - horizon])
            y_val = (closes[train_end + horizon:N] > closes[train_end:N - horizon]).astype(int)
            if len(y_val) > 0:
                wf_preds = clf_wf.predict(X_val)
                wf_accuracy = float(np.mean(wf_preds == y_val))
        except Exception:
            pass

    # 混合模型信心度（60%）與驗證準確率（40%）
    confidence = round(raw_confidence * 0.6 + wf_accuracy * 0.4, 2)
    if confidence < 0.55:
        direction = "neutral"

    return {"price": round(pred_price, 2), "direction": direction, "confidence": confidence}


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
    try:
        url = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=50&page=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = http_requests.get(url, headers=headers, timeout=8)
        items = resp.json().get("data", {}).get("items", [])
        for item in items:
            title = item.get("title", "")
            if stock_name in title or stock_id in title:
                headlines.append(title)
    except Exception:
        pass

    # B1: Claude AI 情緒分析（有新聞時）；無新聞則退回關鍵字法
    avg_score = 0.0
    label = "中性"
    if headlines:
        try:
            news_text = "\n".join(f"- {h}" for h in headlines[:10])
            ai_resp = _anthropic_client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=120,
                system=('你是台灣股市情緒分析師。只回傳 JSON，格式：'
                        '{"score": -1到1之間的浮點數, "label": "正面"或"負面"或"中性"}。'
                        'score > 0 為正面，< 0 為負面，接近 0 為中性。'),
                messages=[{"role": "user", "content":
                           f"股票：{stock_name}({stock_id})\n\n新聞標題：\n{news_text}"}]
            )
            raw = next(b for b in ai_resp.content if b.type == "text").text.strip()
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            avg_score = round(float(parsed.get("score", 0.0)), 2)
            label = parsed.get("label", "中性")
        except Exception as e:
            print(f"Claude 情緒分析失敗 {stock_id}: {e}")
            scores = [analyze_sentiment(h) for h in headlines]
            avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
            label = "正面" if avg_score > 0.1 else "負面" if avg_score < -0.1 else "中性"
    else:
        scores = [analyze_sentiment(h) for h in headlines]
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

    taiex_df = fetch_taiex()
    taiex_change = get_taiex_daily_change()
    winrate_map = get_stock_winrate_map()

    if taiex_change < -0.02:
        print(f"[{today}] ⚠️ 大盤今日下跌 {abs(taiex_change)*100:.1f}%，預測信心度降低")

    for stock in favorites:
        try:
            df = fetch_stock_data(stock["id"])
            if df.empty or len(df) < 40:
                continue
            inst_df = fetch_institutional(stock["id"])
            margin_df = fetch_margin_data(stock["id"])
            news = fetch_news_sentiment(stock["id"], stock["name"])
            sentiment = news.get("score", 0.0)
            current_price = float(df["close"].iloc[-1])
            wr = winrate_map.get(stock["id"])
            result = predict_with_confidence(df, inst_df=inst_df, taiex_df=taiex_df,
                                             margin_df=margin_df, sentiment_score=sentiment,
                                             stock_winrate=wr)
            predicted_price = result["price"]
            direction = result["direction"]
            confidence = result["confidence"]

            conn.execute("""
                INSERT OR IGNORE INTO predictions
                (date, stock_id, stock_name, current_price, predicted_price, predicted_direction, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (today, stock["id"], stock["name"], current_price, predicted_price, direction, confidence))
            conn.commit()

            conn.execute("""
                UPDATE stop_loss SET stop_triggered=0, target_triggered=0
                WHERE stock_id=?
            """, (stock["id"],))
            conn.commit()

            print(f"[{today}] {stock['name']} 預測完成: {current_price} → {predicted_price} ({direction} {confidence*100:.0f}%)")
        except Exception as e:
            print(f"[{today}] {stock['name']} 錯誤: {e}")

    conn.close()
    update_actual_prices()


def update_actual_prices():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, date, stock_id, predicted_direction
        FROM predictions
        WHERE actual_price IS NULL AND date <= date('now', 'localtime')
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


# --- Natural Language Command Parser ---
_anthropic_client = anthropic.Anthropic(max_retries=3, timeout=15.0)
print(f"[INIT] ANTHROPIC_API_KEY loaded: {'YES' if os.environ.get('ANTHROPIC_API_KEY') else 'NO'}")

# 規則比對備援
_RULE_MAP = [
    (["預測", "分析", "跑一下", "預估", "幫我看"], "predict"),
    (["新聞", "消息", "報導", "情緒"], "news"),
    (["加入", "追蹤", "新增", "訂閱"], "add"),
    (["刪除", "移除", "取消", "退出"], "remove"),
    (["清單", "我的股票", "自選", "列表"], "list"),
    (["停損", "檢查", "價格", "虧損"], "check"),
    (["報告", "今天", "今日", "結果"], "report"),
    (["說明", "怎麼用", "指令", "幫助"], "help"),
]
_STOCK_RE = __import__("re").compile(r"(?<!\d)(\d{4,6})(?!\d)")

# 股票名稱/簡稱 → 代碼對照表（啟動時從 TWSE/TPEx 動態載入）
def _build_name_map() -> tuple:
    name_to_code = {}
    code_to_name = {}
    sources = [
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",    # 上市
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", # 上櫃
    ]
    for url in sources:
        try:
            resp = http_requests.get(url, timeout=10)
            resp.raise_for_status()
            for row in resp.json():
                code = (row.get("公司代號") or row.get("SecuritiesCompanyCode") or "").strip()
                short = (row.get("公司簡稱") or row.get("CompanyAbbreviation") or "").strip()
                full = (row.get("公司名稱") or row.get("CompanyName") or "").strip()
                if not code:
                    continue
                if short:
                    name_to_code[short] = code
                    code_to_name[code] = short   # 以簡稱為顯示名稱
                if full and full != short:
                    name_to_code[full] = code
        except Exception as e:
            print(f"[INIT] 無法載入股票對照表 {url}: {e}")
    print(f"[INIT] 股票名稱對照表載入 {len(name_to_code)} 筆")
    return name_to_code, code_to_name

_NAME_MAP, _CODE_TO_NAME = _build_name_map()

def _rule_parse(text: str) -> dict:
    m = _STOCK_RE.search(text)
    stock_id = m.group(1) if m else None
    if not stock_id:
        # 名稱對照表（長名稱優先，避免短名稱誤中）
        for name in sorted(_NAME_MAP, key=len, reverse=True):
            if name in text:
                stock_id = _NAME_MAP[name]
                break
    if not stock_id:
        for s in load_favorites():
            if s["name"] in text:
                stock_id = s["id"]
                break
    for words, cmd in _RULE_MAP:
        if any(w in text for w in words):
            return {"command": cmd, "stock_id": stock_id}
    return {"command": "unknown", "stock_id": None}

def parse_natural_language_command(text: str) -> dict:
    try:
        favorites = load_favorites()
        stock_list = "、".join([f"{s['name']}({s['id']})" for s in favorites]) or "（無自選股）"
        response = _anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            system=f"""你是台灣股票機器人指令解析器。只回傳 JSON，不要其他文字。

自選股: {stock_list}

指令:
- predict: 預測/分析（含股票名或代碼表示單股預測）
- report: 今日報告
- check: 檢查停損
- news: 新聞（需 stock_id）
- add: 加入自選股（需 stock_id）
- remove: 刪除自選股（需 stock_id）
- list: 清單
- help: 說明
- unknown: 無法識別

stock_id: 台灣股票代碼數字（如2330），名稱轉代碼（台積電→2330，台灣大哥大→3045），無則 null。

格式: {{"command":"...","stock_id":null}}""",
            messages=[{"role": "user", "content": text}]
        )
        text_block = next(b for b in response.content if b.type == "text")
        raw = text_block.text.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception as e:
        print(f"NL parse error: {e}，改用規則比對")
        return _rule_parse(text)


# --- Telegram ---
def _lookup_stock_name(stock_id: str) -> str:
    name = _CODE_TO_NAME.get(stock_id)
    if name:
        return name
    try:
        api = DataLoader()
        df = api.taiwan_stock_info()
        row = df[df["stock_id"] == stock_id]
        if not row.empty:
            n = str(row["stock_name"].iloc[0]).strip()
            _CODE_TO_NAME[stock_id] = n
            return n
    except Exception:
        pass
    return stock_id


def predict_and_report_single(stock_id: str):
    today = datetime.now().strftime("%Y-%m-%d")
    favorites = load_favorites()
    fav = next((s for s in favorites if s["id"] == stock_id), None)
    stock_name = fav["name"] if fav else _lookup_stock_name(stock_id)
    try:
        send_telegram(f"⏳ 分析 {stock_name}（{stock_id}）中，請稍候...")
        df = fetch_stock_data(stock_id)
        if df.empty or len(df) < 40:
            send_telegram(f"❌ {stock_name}（{stock_id}）資料不足，無法預測")
            return
        taiex_df = fetch_taiex()
        taiex_change = get_taiex_daily_change()
        inst_df = fetch_institutional(stock_id)
        margin_df = fetch_margin_data(stock_id)
        news = fetch_news_sentiment(stock_id, stock_name)
        sentiment = news.get("score", 0.0)
        current_price = float(df["close"].iloc[-1])
        wr_map = get_stock_winrate_map()
        result = predict_with_confidence(df, inst_df=inst_df, taiex_df=taiex_df,
                                         margin_df=margin_df, sentiment_score=sentiment,
                                         stock_winrate=wr_map.get(stock_id))
        predicted_price = result["price"]
        direction = result["direction"]
        confidence = result["confidence"]

        pos = calc_position_size(confidence, direction,
                                  winrate=wr_map.get(stock_id), taiex_change=taiex_change)

        if direction == "neutral":
            arrow, dir_text = "⚠️", f"方向不明（信心度 {confidence*100:.0f}%）"
        else:
            arrow = "↑" if direction == "up" else "↓"
            dir_text = f"{'看漲' if direction == 'up' else '看跌'}（信心度 {confidence*100:.0f}%）"

        pos_text = f"\n💼 建議倉位：{pos['label']}"
        if pos["reason"]:
            pos_text += f"（{pos['reason']}）"
        sentiment_text = f"\n📰 新聞情緒：{news.get('label', '')}" if news.get("label") else ""
        market_warning = f"\n⚠️ 大盤今跌 {abs(taiex_change)*100:.1f}%，請謹慎" if taiex_change < -0.02 else ""
        send_telegram(
            f"📊 <b>{stock_name}（{stock_id}）預測</b>\n\n"
            f"目前價格：{current_price}\n"
            f"預測方向：{arrow} {dir_text}\n"
            f"預測價格：<b>{predicted_price}</b>{pos_text}{sentiment_text}{market_warning}\n\n"
            f"🤖 AI 預測僅供參考，投資請謹慎"
        )
    except Exception as e:
        send_telegram(f"❌ 分析 {stock_name}（{stock_id}）失敗：{e}")


def send_telegram(message: str):
    cfg = load_config()["telegram"]
    url = f"https://api.telegram.org/bot{cfg['token']}/sendMessage"
    http_requests.post(url, json={"chat_id": cfg["chat_id"], "text": message, "parse_mode": "HTML"})


def send_single_stock_report(stock_id: str):
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT p.stock_name, p.stock_id, p.current_price, p.predicted_price, p.predicted_direction,
               n.label, n.sentiment_score
        FROM predictions p
        LEFT JOIN news_cache n ON p.stock_id = n.stock_id
        WHERE p.date = ? AND p.stock_id = ?
    """, (today, stock_id)).fetchall()
    conn.close()
    if not rows:
        send_telegram(f"⚠️ 今日尚無 {stock_id} 的預測資料")
        return
    r = rows[0]
    arrow = "↑" if r[4] == "up" else "↓"
    sentiment = f"\n📰 新聞情緒：{r[5]}" if r[5] else ""
    send_telegram(
        f"📊 <b>{r[0]}（{r[1]}）預測</b>\n\n"
        f"目前價格：{r[2]}\n"
        f"預測方向：{arrow} {'看漲' if r[4] == 'up' else '看跌'}\n"
        f"預測價格：<b>{r[3]}</b>{sentiment}\n\n"
        f"🤖 AI 預測僅供參考，投資請謹慎"
    )


def send_morning_report():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT p.stock_name, p.stock_id, p.current_price, p.predicted_price, p.predicted_direction,
               n.label, p.confidence
        FROM predictions p
        LEFT JOIN news_cache n ON p.stock_id = n.stock_id
        WHERE p.date = ?
        ORDER BY p.confidence DESC NULLS LAST, p.stock_id
    """, (today,)).fetchall()
    conn.close()

    if not rows:
        send_telegram(f"📊 <b>{today} 股票預測</b>\n\n今日尚無預測資料，請先執行預測。")
        return

    taiex_change = get_taiex_daily_change()
    wr_map = get_stock_winrate_map()
    up_list = [r for r in rows if r[4] == "up"]
    down_list = [r for r in rows if r[4] == "down"]
    neutral_list = [r for r in rows if r[4] == "neutral"]

    lines = [f"📊 <b>{today} 每日股票預測報告</b>\n"]
    if taiex_change < -0.02:
        lines.append(f"⚠️ 大盤今跌 {abs(taiex_change)*100:.1f}%，預測信心度參考性降低\n")
    lines.append(f"看漲 {len(up_list)} 支 ｜ 看跌 {len(down_list)} 支 ｜ 中性 {len(neutral_list)} 支\n")
    lines.append("─────────────────────")

    def fmt_row(r, arrow):
        conf = r[6] or 0.0
        wr = wr_map.get(r[1])
        pos = calc_position_size(conf, r[4], winrate=wr, taiex_change=taiex_change)
        conf_str = f" [{conf*100:.0f}%]" if conf else ""
        pos_str = f" 【{pos['label']}】" if pos["ratio"] > 0 else " 【不進場】"
        sentiment = f" {r[5]}" if r[5] else ""
        return f"  {arrow} {r[0]}({r[1]})  {r[2]} → <b>{r[3]}</b>{conf_str}{pos_str}{sentiment}"

    lines.append("\n📈 <b>看漲</b>")
    for r in up_list:
        lines.append(fmt_row(r, "↑"))

    lines.append("\n📉 <b>看跌</b>")
    for r in down_list:
        lines.append(fmt_row(r, "↓"))

    if neutral_list:
        lines.append("\n⚠️ <b>方向不明</b>")
        for r in neutral_list:
            lines.append(fmt_row(r, "—"))

    lines.append("\n─────────────────────")
    lines.append("🤖 AI 預測僅供參考，投資請謹慎")
    send_telegram("\n".join(lines))


def send_accuracy_report():
    """14:00 收盤後：先更新實際價格，再發送昨日預測準確度報告。"""
    update_actual_prices()
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    # 取昨日或今日最近一批有驗證結果的預測
    rows = conn.execute("""
        SELECT stock_name, stock_id, predicted_direction, actual_direction,
               is_correct, current_price, actual_price
        FROM predictions
        WHERE date IN (?, ?)
          AND is_correct IS NOT NULL
        ORDER BY date DESC, stock_id
    """, (today, yesterday)).fetchall()

    overall = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END)
        FROM predictions
        WHERE date >= date('now','localtime','-30 days')
          AND is_correct IS NOT NULL
    """).fetchone()
    conn.close()

    if not rows:
        send_telegram("📊 <b>收盤準確度報告</b>\n\n尚無已驗證的預測結果，資料可能仍在更新中。")
        return

    total_judged = len(rows)
    correct = sum(1 for r in rows if r[4] == 1)
    today_rate = round(correct / total_judged * 100, 1) if total_judged else 0

    ov_total = overall[0] or 0
    ov_correct = overall[1] or 0
    ov_rate = round(ov_correct / ov_total * 100, 1) if ov_total > 0 else None

    lines = [f"📊 <b>今日收盤預測準確度報告</b>\n"]
    lines.append(f"今日命中：{correct}/{total_judged}（{today_rate}%）")
    if ov_rate is not None:
        lines.append(f"近 30 日勝率：{ov_rate}%（{ov_correct}/{ov_total} 筆）\n")
    lines.append("─────────────────────")

    correct_rows = [r for r in rows if r[4] == 1]
    wrong_rows   = [r for r in rows if r[4] == 0]

    if correct_rows:
        lines.append("\n✅ <b>預測正確</b>")
        for r in correct_rows:
            arrow = "↑" if r[2] == "up" else "↓"
            chg = f"{r[6]:.0f}" if r[6] else "?"
            lines.append(f"  {arrow} {r[0]}({r[1]})  實際:{chg}")

    if wrong_rows:
        lines.append("\n❌ <b>預測錯誤</b>")
        for r in wrong_rows:
            pred_arrow = "↑" if r[2] == "up" else "↓"
            actual_arrow = "↑" if r[3] == "up" else "↓"
            lines.append(f"  {pred_arrow}→{actual_arrow} {r[0]}({r[1]})")

    lines.append("\n─────────────────────")
    lines.append("🤖 持續學習中，勝率將逐步提升")
    send_telegram("\n".join(lines))


# --- Telegram command handler ---
_last_update_id = 0

HELP_TEXT = """📋 <b>可用指令</b>

/預測 — 立即執行預測並發送報告
/報告 — 發送今日預測報告
/勝率 — 今日收盤準確度報告
/檢查 — 立即檢查停損價
/新聞 2330 — 查詢股票新聞情緒
/加入 2330 — 新增自選股
/刪除 2330 — 刪除自選股
/清單 — 顯示目前自選股
/說明 — 顯示此說明"""


def handle_telegram_commands():
    global _last_update_id
    cfg = load_config()["telegram"]
    token = cfg["token"]
    allowed_chat = str(cfg["chat_id"])

    # 啟動時清空所有待處理訊息
    try:
        resp = http_requests.get(f"https://api.telegram.org/bot{token}/getUpdates?timeout=0", timeout=10)
        updates = resp.json().get("result", [])
        if updates:
            _last_update_id = updates[-1]["update_id"]
            # 告知 Telegram 已處理到此 update_id
            http_requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates?offset={_last_update_id + 1}&timeout=0",
                timeout=10
            )
    except Exception:
        pass

    while True:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates?offset={_last_update_id + 1}&timeout=30"
            resp = http_requests.get(url, timeout=35)
            updates = resp.json().get("result", [])

            for update in updates:
                _last_update_id = update["update_id"]
                msg = update.get("message", {})
                chat_id = str(msg.get("chat", {}).get("id", ""))
                text = msg.get("text", "").strip()

                if chat_id != allowed_chat or not text:
                    continue

                parts = text.split()
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ["/預測", "/predict"]:
                    send_telegram("⏳ 開始執行預測，請稍候...")
                    run_daily_predictions()
                    send_morning_report()

                elif cmd in ["/報告", "/report"]:
                    send_morning_report()

                elif cmd in ["/檢查", "/check"]:
                    send_telegram("🔍 檢查停損價中...")
                    check_stop_loss()
                    send_telegram("✅ 停損檢查完成")

                elif cmd in ["/新聞", "/news"] and arg:
                    stocks = load_favorites()
                    stock = next((s for s in stocks if s["id"] == arg), {"name": arg})
                    result = fetch_news_sentiment(arg, stock["name"])
                    headlines = "\n".join(f"• {h}" for h in result["headlines"]) or "無相關新聞"
                    send_telegram(f"📰 <b>{stock['name']}({arg}) 新聞情緒：{result['label']}</b>\n\n{headlines}")

                elif cmd in ["/加入", "/add"] and arg:
                    try:
                        api = DataLoader()
                        df = api.taiwan_stock_info()
                        row = df[df["stock_id"] == arg]
                        name = str(row["stock_name"].iloc[0]) if not row.empty else arg
                        stocks = load_favorites()
                        if any(s["id"] == arg for s in stocks):
                            send_telegram(f"⚠️ {name}({arg}) 已在清單中")
                        else:
                            stocks.append({"id": arg, "name": name})
                            save_favorites(stocks)
                            send_telegram(f"✅ 已加入：{name}({arg})")
                    except Exception as e:
                        send_telegram(f"❌ 加入失敗：{e}")

                elif cmd in ["/刪除", "/remove"] and arg:
                    stocks = load_favorites()
                    found = next((s for s in stocks if s["id"] == arg), None)
                    if found:
                        save_favorites([s for s in stocks if s["id"] != arg])
                        send_telegram(f"✅ 已刪除：{found['name']}({arg})")
                    else:
                        send_telegram(f"⚠️ 找不到股票代碼 {arg}")

                elif cmd in ["/清單", "/list"]:
                    stocks = load_favorites()
                    lines = [f"• {s['name']}（{s['id']}）" for s in stocks]
                    send_telegram(f"⭐ <b>自選股清單（{len(stocks)} 支）</b>\n\n" + "\n".join(lines))

                elif cmd in ["/勝率", "/accuracy"]:
                    send_accuracy_report()

                elif cmd in ["/說明", "/help"]:
                    send_telegram(HELP_TEXT)

                else:
                    # 自然語言解析
                    parsed = parse_natural_language_command(text)
                    nl_cmd = parsed.get("command", "unknown")
                    nl_arg = parsed.get("stock_id") or ""

                    if nl_cmd == "predict":
                        if nl_arg:
                            predict_and_report_single(nl_arg)
                        else:
                            send_telegram("⏳ 開始執行預測，請稍候...")
                            run_daily_predictions()
                            send_morning_report()

                    elif nl_cmd == "report":
                        send_morning_report()

                    elif nl_cmd == "check":
                        send_telegram("🔍 檢查停損價中...")
                        check_stop_loss()
                        send_telegram("✅ 停損檢查完成")

                    elif nl_cmd == "news" and nl_arg:
                        stocks = load_favorites()
                        stock = next((s for s in stocks if s["id"] == nl_arg), {"name": nl_arg})
                        result = fetch_news_sentiment(nl_arg, stock["name"])
                        headlines = "\n".join(f"• {h}" for h in result["headlines"]) or "無相關新聞"
                        send_telegram(f"📰 <b>{stock['name']}({nl_arg}) 新聞情緒：{result['label']}</b>\n\n{headlines}")

                    elif nl_cmd == "add" and nl_arg:
                        try:
                            api = DataLoader()
                            df = api.taiwan_stock_info()
                            row = df[df["stock_id"] == nl_arg]
                            name = str(row["stock_name"].iloc[0]) if not row.empty else nl_arg
                            stocks = load_favorites()
                            if any(s["id"] == nl_arg for s in stocks):
                                send_telegram(f"⚠️ {name}({nl_arg}) 已在清單中")
                            else:
                                stocks.append({"id": nl_arg, "name": name})
                                save_favorites(stocks)
                                send_telegram(f"✅ 已加入：{name}({nl_arg})")
                        except Exception as e:
                            send_telegram(f"❌ 加入失敗：{e}")

                    elif nl_cmd == "remove" and nl_arg:
                        stocks = load_favorites()
                        found = next((s for s in stocks if s["id"] == nl_arg), None)
                        if found:
                            save_favorites([s for s in stocks if s["id"] != nl_arg])
                            send_telegram(f"✅ 已刪除：{found['name']}({nl_arg})")
                        else:
                            send_telegram(f"⚠️ 找不到股票代碼 {nl_arg}")

                    elif nl_cmd == "list":
                        stocks = load_favorites()
                        lines = [f"• {s['name']}（{s['id']}）" for s in stocks]
                        send_telegram(f"⭐ <b>自選股清單（{len(stocks)} 支）</b>\n\n" + "\n".join(lines))

                    elif nl_cmd == "help":
                        send_telegram(HELP_TEXT)

                    else:
                        send_telegram(f"❓ 無法識別您的指令\n\n輸入 /說明 查看所有指令")

        except Exception as e:
            print(f"Telegram polling error: {e}")
            import time
            time.sleep(5)


# --- Scheduler ---
def morning_routine():
    """08:00 產預測 → 發晨報"""
    run_daily_predictions()
    send_morning_report()

scheduler = BackgroundScheduler()
scheduler.add_job(morning_routine, "cron", hour=8, minute=0)          # 08:00 產預測+晨報
scheduler.add_job(send_accuracy_report, "cron", hour=14, minute=0)    # 14:00 收盤準確度
scheduler.add_job(check_stop_loss, "cron", minute="*/30")             # 每 30 分鐘停損檢查
# 每週日 01:00 深度重訓
scheduler.add_job(morning_routine, "cron", day_of_week="sun", hour=1, minute=0)
scheduler.start()

# 啟動 Telegram 指令監聽（設定 DISABLE_TELEGRAM=1 可停用，用於本機開發）
if not os.environ.get("DISABLE_TELEGRAM"):
    threading.Thread(target=handle_telegram_commands, daemon=True).start()
else:
    print("[INIT] Telegram polling disabled (DISABLE_TELEGRAM=1)")


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
        inst_df = fetch_institutional(req.stock_id)
        margin_df = fetch_margin_data(req.stock_id)
        taiex_df = fetch_taiex()
        news = fetch_news_sentiment(req.stock_id, req.stock_id)
        sentiment = news.get("score", 0.0)
        df = calc_indicators(df)
        recent = df.tail(60).copy()
        predictions = predict_xgb(df, inst_df=inst_df, taiex_df=taiex_df, sentiment_score=sentiment)
        wr_map = get_stock_winrate_map()
        clf_result = predict_with_confidence(df, inst_df=inst_df, taiex_df=taiex_df,
                                              margin_df=margin_df, sentiment_score=sentiment,
                                              stock_winrate=wr_map.get(req.stock_id))
        direction = clf_result["direction"]
        confidence = clf_result["confidence"]
        taiex_change = get_taiex_daily_change()
        pos = calc_position_size(confidence, direction,
                                 winrate=wr_map.get(req.stock_id), taiex_change=taiex_change)
        last_date = pd.to_datetime(df["date"].iloc[-1])
        future_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(5)]
        close = recent["close"].tolist()
        if direction == "neutral":
            trend_text = f"⚠️ 方向不明（{confidence*100:.0f}%）"
        else:
            arrow = "↑" if direction == "up" else "↓"
            trend_text = f"{arrow} {'看漲' if direction == 'up' else '看跌'}（{confidence*100:.0f}%）"
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
            "trend": trend_text,
            "confidence": confidence,
            "direction": direction,
            "position_label": pos["label"],
            "position_ratio": pos["ratio"],
            "position_reason": pos["reason"],
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
    threading.Thread(target=morning_routine, daemon=True).start()
    return {"status": "ok", "message": "預測執行中，完成後將發送 Telegram"}


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


@app.post("/update-actuals")
async def trigger_update_actuals():
    threading.Thread(target=update_actual_prices, daemon=True).start()
    return {"status": "started"}
