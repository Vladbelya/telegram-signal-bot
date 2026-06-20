import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# --- TELEGRAM BOT CONFIG ---
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# ==========================================
# --- TRADING CONFIG ---
# ==========================================
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TRADE_LOG_FILE = "trade_log.json"
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01  # 1% of equity per trade

# ==========================================
# --- STRATEGY CONFIGS (from backtests) ---
# ==========================================
# Each strategy monitors a different timeframe for wick volume anomalies.
# Parameters are the best-performing combinations from grid search backtests.
#
# Signal Logic:
#   SHORT: up_wick_total >= threshold AND up_wick_buy >= up_wick_sell * imbalance_ratio
#   LONG:  lo_wick_total >= threshold AND lo_wick_sell >= lo_wick_buy * imbalance_ratio
#   SL:    behind high/low + sl_buffer_pct
#   TP:    entry ± (dist_to_sl * rr)

STRATEGIES = {
    # ===== BTCUSDT =====
    "BTC_15M": {
        "name": "BTC 15M",
        "symbol": "BTCUSDT",
        "tf_str": "15min",
        "tf_minutes": 15,
        "top_percent": 3,
        "imbalance_ratio": 3.0,
        "rr": 3.0,
        "sl_buffer_pct": 0.0005,
    },
    "BTC_30M": {
        "name": "BTC 30M",
        "symbol": "BTCUSDT",
        "tf_str": "30min",
        "tf_minutes": 30,
        "top_percent": 4,
        "imbalance_ratio": 2.0,
        "rr": 2.0,
        "sl_buffer_pct": 0.0005,
    },
    "BTC_1H": {
        "name": "BTC 1H",
        "symbol": "BTCUSDT",
        "tf_str": "1h",
        "tf_minutes": 60,
        "top_percent": 9,
        "imbalance_ratio": 2.0,
        "rr": 2.5,
        "sl_buffer_pct": 0.001,
    },
    "BTC_4H": {
        "name": "BTC 4H",
        "symbol": "BTCUSDT",
        "tf_str": "4h",
        "tf_minutes": 240,
        "top_percent": 8,
        "imbalance_ratio": 1.6,
        "rr": 3.0,
        "sl_buffer_pct": 0.0015,
    },
    # ===== ETHUSDT =====
    "ETH_15M": {
        "name": "ETH 15M",
        "symbol": "ETHUSDT",
        "tf_str": "15min",
        "tf_minutes": 15,
        "top_percent": 10,
        "imbalance_ratio": 3.0,
        "rr": 3.0,
        "sl_buffer_pct": 0.0015,
    },
    "ETH_30M": {
        "name": "ETH 30M",
        "symbol": "ETHUSDT",
        "tf_str": "30min",
        "tf_minutes": 30,
        "top_percent": 7,
        "imbalance_ratio": 1.6,
        "rr": 2.5,
        "sl_buffer_pct": 0.0005,
    },
    "ETH_1H": {
        "name": "ETH 1H",
        "symbol": "ETHUSDT",
        "tf_str": "1h",
        "tf_minutes": 60,
        "top_percent": 9,
        "imbalance_ratio": 1.6,
        "rr": 3.0,
        "sl_buffer_pct": 0.0015,
    },
}
