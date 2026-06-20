"""
BTC 5M Scalp Backtest — FAST VERSION (numpy-based engine)
Strategy: Wick Volume Imbalance on 5-minute candles
"""

import requests
import pandas as pd
import numpy as np
import datetime
import time
import os
import sys

# ==========================================
# --- CONFIGURATION ---
# ==========================================
SYMBOL = "BTCUSDT"
FILENAME = "btc_2025_data.csv"
DAYS_TO_FETCH = 450

RATIOS = {
    '1.25x (80%)': 0.80,
    '1.5x (67%)':  0.67,
    '2x (50%)':    0.50,
    '2.5x (40%)':  0.40,
    '3x (33%)':    0.33,
    '4x (25%)':    0.25,
    '5x (20%)':    0.20,
    '6x (17%)':    0.17,
}

# Volume thresholds (~2x previous round, now 1200-5000)
VOLUME_THRESHOLDS = {
    '>1200': 1200,
    '>1500': 1500,
    '>1800': 1800,
    '>2200': 2200,
    '>2800': 2800,
    '>3500': 3500,
    '>5000': 5000,
}

SL_PARAMETERS = [20, 30, 50, 75]

# Multiple RR values to test
RR_VALUES = [1.2, 1.6, 2.0, 2.5]

# Binance Futures commission: 0.04% taker x2 (entry + exit) = 0.08% per trade
# Applied as fraction of position size (equity used as 1x position proxy)
COMMISSION = 0.0008  # 0.08% per round-trip
RISK_PER_TRADE = 0.05  # 5% of equity risked per trade

# ==========================================
# --- DOWNLOADER ---
# ==========================================
class BinanceKlines:
    def __init__(self, api_key=None):
        self.base_url = "https://fapi.binance.com"
        self.headers = {'X-MBX-APIKEY': api_key} if api_key else {}

    def download_history(self, symbol, days):
        end_dt = datetime.datetime.now(datetime.timezone.utc)
        start_dt = end_dt - datetime.timedelta(days=days)
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        all_klines = []
        current_start = start_ts
        total_minutes = days * 1440

        print(f"[*] Downloading {symbol} ({days} days)...")
        t_start = time.time()

        while current_start < end_ts:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {"symbol": symbol, "interval": "1m", "startTime": current_start, "limit": 1500}
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=10)
                if r.status_code in (418, 429):
                    time.sleep(60); continue
                if r.status_code != 200:
                    time.sleep(2); continue
                data = r.json()
                if not data: break
                for k in data:
                    all_klines.append([k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9])])
                current_start = data[-1][0] + 60000
                if len(all_klines) % 10000 < 1500:
                    pct = min(len(all_klines) / total_minutes * 100, 100)
                    bar = '#' * int(30 * pct / 100) + '-' * (30 - int(30 * pct / 100))
                    sys.stdout.write(f"\r    [{bar}] {pct:.0f}% | {len(all_klines):,} candles  ")
                    sys.stdout.flush()
                time.sleep(0.2)
            except Exception as e:
                print(f"\n[ERROR] {e}"); time.sleep(2)

        print(f"\n[OK] Downloaded {len(all_klines):,} candles in {time.time()-t_start:.0f}s")
        df = pd.DataFrame(all_klines, columns=['ts','open','high','low','close','vol','taker_buy_vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

# ==========================================
# --- DATA PROCESSING (1M -> 5M) ---
# ==========================================
def process_data(df_1m):
    print("[*] Resampling to 5M and calculating Wick Deltas...")
    t0 = time.time()

    df_1m = df_1m.copy()
    df_1m['sell_vol'] = df_1m['vol'] - df_1m['taker_buy_vol']

    df_5m = df_1m.set_index('ts').resample('5min').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
    ).dropna().reset_index()

    df_1m['h_ts'] = df_1m['ts'].dt.floor('5min')
    merged = pd.merge(df_1m, df_5m[['ts','open','close']],
                      left_on='h_ts', right_on='ts', suffixes=('','_5m'))

    merged['wick_top'] = merged[['open_5m','close_5m']].max(axis=1)
    merged['wick_bot'] = merged[['open_5m','close_5m']].min(axis=1)

    mask_up = merged['close'] >= merged['wick_top']
    upper_buys  = merged[mask_up].groupby('h_ts')['taker_buy_vol'].sum()
    upper_sells = merged[mask_up].groupby('h_ts')['sell_vol'].sum()

    mask_lo = merged['close'] <= merged['wick_bot']
    lower_buys  = merged[mask_lo].groupby('h_ts')['taker_buy_vol'].sum()
    lower_sells = merged[mask_lo].groupby('h_ts')['sell_vol'].sum()

    df_5m = df_5m.set_index('ts')
    df_5m['up_wick_buy']  = upper_buys
    df_5m['up_wick_sell'] = upper_sells
    df_5m['lo_wick_buy']  = lower_buys
    df_5m['lo_wick_sell'] = lower_sells
    df_5m.fillna(0, inplace=True)

    print(f"[*] Processed {len(df_5m):,} 5M candles in {time.time()-t0:.1f}s")
    return df_5m.reset_index()

# ==========================================
# --- FAST BACKTEST ENGINE (numpy) ---
# ==========================================
def run_backtest_fast(df, threshold, ratio, sl_pips, rr):
    """
    Backtest with fixed 5% risk per trade.
    risk_amount = equity * 5%
    position_size (BTC) = risk_amount / sl_pips
    win PnL  = position_size * sl_pips * rr
    loss PnL = -risk_amount
    commission = position_size * entry_price * COMMISSION (0.08% of notional)
    """
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    up_buy  = df['up_wick_buy'].values
    up_sell = df['up_wick_sell'].values
    lo_buy  = df['lo_wick_buy'].values
    lo_sell = df['lo_wick_sell'].values

    n = len(df)
    equity = 10000.0
    trades_count = 0
    wins = 0

    in_pos = False
    pos_type = 0  # 1=Long, -1=Short
    entry = sl = tp = pos_size = 0.0

    for i in range(n):
        if equity <= 0:
            break
        if in_pos:
            if pos_type == -1:  # Short
                if highs[i] >= sl:
                    commission = pos_size * entry * COMMISSION
                    equity -= (pos_size * sl_pips) + commission   # loss
                    trades_count += 1
                    in_pos = False
                elif lows[i] <= tp:
                    commission = pos_size * entry * COMMISSION
                    equity += (pos_size * sl_pips * rr) - commission  # win
                    trades_count += 1
                    wins += 1
                    in_pos = False
            else:  # Long
                if lows[i] <= sl:
                    commission = pos_size * entry * COMMISSION
                    equity -= (pos_size * sl_pips) + commission   # loss
                    trades_count += 1
                    in_pos = False
                elif highs[i] >= tp:
                    commission = pos_size * entry * COMMISSION
                    equity += (pos_size * sl_pips * rr) - commission  # win
                    trades_count += 1
                    wins += 1
                    in_pos = False
            continue

        # SHORT signal
        if up_buy[i] >= threshold and up_sell[i] <= up_buy[i] * ratio:
            in_pos = True; pos_type = -1
            entry = closes[i]
            sl = highs[i] + sl_pips
            tp = entry - sl_pips * rr
            risk_amount = equity * RISK_PER_TRADE
            pos_size = risk_amount / sl_pips  # BTC size

        # LONG signal
        elif lo_sell[i] >= threshold and lo_buy[i] <= lo_sell[i] * ratio:
            in_pos = True; pos_type = 1
            entry = closes[i]
            sl = lows[i] - sl_pips
            tp = entry + sl_pips * rr
            risk_amount = equity * RISK_PER_TRADE
            pos_size = risk_amount / sl_pips  # BTC size

    return trades_count, wins, equity


# ==========================================
# --- REPORT ---
# ==========================================
def generate_report(results):
    sorted_res = sorted(results, key=lambda x: x['final_equity'], reverse=True)

    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>BTC 5M Scalp Backtest</title>
    <style>
        body{font-family:sans-serif;padding:30px;background:#0d1117;color:#e6edf3;}
        h1{color:#f7931a;font-size:28px;} h2{color:#58a6ff;margin-top:32px;font-size:20px;}
        p{color:#8b949e;}
        table{border-collapse:collapse;width:100%;margin-top:16px;}
        th,td{border:1px solid #30363d;padding:10px 14px;text-align:left;font-size:14px;}
        tr:nth-child(even){background:#161b22;}
        th{background:#1c2128;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;}
        .profit{color:#3fb950;font-weight:700;}
        .loss{color:#f85149;font-weight:700;}
        .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;}
        .b-green{background:rgba(63,185,80,.15);color:#3fb950;}
        .b-orange{background:rgba(247,147,26,.15);color:#f7931a;}
        .b-blue{background:rgba(88,166,255,.15);color:#58a6ff;}
        .b-purple{background:rgba(188,140,255,.15);color:#bc8cff;}
        .top1{background:rgba(247,147,26,.06);}
    </style></head><body>"""

    html += "<h1>BTCUSDT 5M Scalp Backtest (2025) | With Binance Commission (0.08%/trade)</h1>"
    html += f"<p>Grid search: {len(results)} combinations | Timeframe: 5M | Capital: $10,000 | Commission: 0.08% per round-trip | Data: 2025</p>"
    html += "<h2>Top 50 Configurations</h2>"
    html += ("<table><tr><th>#</th><th>Volume Threshold</th><th>Imbalance</th>"
             "<th>SL ($)</th><th>RR</th><th>Trades</th><th>Win Rate</th>"
             "<th>Final Equity</th><th>P&L ($)</th><th>P&L (%)</th></tr>")

    for i, res in enumerate(sorted_res[:50]):
        wr = (res['wins'] / res['trades'] * 100) if res['trades'] > 0 else 0
        pnl = res['final_equity'] - 10000
        pnl_pct = pnl / 10000 * 100
        pnl_cls = 'profit' if pnl >= 0 else 'loss'
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        pnl_pct_str = f"+{pnl_pct:.1f}%" if pnl_pct >= 0 else f"{pnl_pct:.1f}%"
        row_cls = ' class="top1"' if i == 0 else ''
        rank = '#1' if i == 0 else '#2' if i == 1 else '#3' if i == 2 else str(i+1)
        html += (f"<tr{row_cls}><td><b>{rank}</b></td>"
                 f"<td>{res['threshold_label']}</td>"
                 f"<td><span class='badge b-orange'>{res['ratio']}</span></td>"
                 f"<td>${res['sl']}</td>"
                 f"<td><span class='badge b-purple'>1:{res['rr']}</span></td>"
                 f"<td>{res['trades']}</td>"
                 f"<td><span class='badge b-green'>{wr:.1f}%</span></td>"
                 f"<td><b>${res['final_equity']:.2f}</b></td>"
                 f"<td class='{pnl_cls}'>{pnl_str}</td>"
                 f"<td class='{pnl_cls}'><b>{pnl_pct_str}</b></td></tr>")

    html += "</table></body></html>"

    with open("btc_5m_backtest_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("[DONE] Report: btc_5m_backtest_report.html")

# ==========================================
# --- MAIN ---
# ==========================================
if __name__ == "__main__":
    # 1. Load data
    if os.path.exists(FILENAME):
        print(f"[*] Loading {FILENAME}...")
        df_1m = pd.read_csv(FILENAME)
        df_1m['ts'] = pd.to_datetime(df_1m['ts'])
    else:
        API_KEY = "7n8HrqdOnOc1RfpRsle2QEFwBUerJKnxOpW09yjaxU4eocUzwjpjJIEGtEJyKebs"
        client = BinanceKlines(api_key=API_KEY)
        df_1m = client.download_history(SYMBOL, DAYS_TO_FETCH)
        if not df_1m.empty:
            df_1m.to_csv(FILENAME, index=False)

    # 2. Filter 2025
    df_1m = df_1m[(df_1m['ts'] >= '2025-01-01') & (df_1m['ts'] <= '2025-12-31')]
    if df_1m.empty:
        print("[ERROR] No data for 2025!"); exit()
    print(f"[*] Loaded {len(df_1m):,} 1M candles for 2025")

    # 3. Process to 5M
    df_5m = process_data(df_1m)

    # 4. Show reference percentiles
    up_vols = df_5m[df_5m['up_wick_buy'] > 1]['up_wick_buy']
    lo_vols = df_5m[df_5m['lo_wick_sell'] > 1]['lo_wick_sell']
    all_vols = pd.concat([up_vols, lo_vols])
    print(f"\n[*] Reference percentiles (for context):")
    for p in [90, 93, 95, 97, 98, 99]:
        print(f"    Top {100-p}%: {all_vols.quantile(p/100):.1f}")

    print(f"\n[*] Testing fixed thresholds: {list(VOLUME_THRESHOLDS.keys())}")

    # 5. Grid Search: threshold x ratio x SL x RR
    results = []
    total_iter = len(VOLUME_THRESHOLDS) * len(RATIOS) * len(SL_PARAMETERS) * len(RR_VALUES)
    print(f"[*] Grid Search: {total_iter} combinations (numpy engine)...")
    counter = 0
    bar_len = 40
    t_gs = time.time()

    for thr_label, thr_val in VOLUME_THRESHOLDS.items():
        for r_name, r_val in RATIOS.items():
            for sl in SL_PARAMETERS:
                for rr in RR_VALUES:
                    counter += 1
                    count, wins, eq = run_backtest_fast(df_5m, thr_val, r_val, sl, rr)
                    results.append({
                        'threshold_label': thr_label,
                        'threshold_val': thr_val,
                        'ratio': r_name,
                        'sl': sl,
                        'rr': rr,
                        'trades': count,
                        'wins': wins,
                        'final_equity': eq
                    })
                    pct = counter / total_iter * 100
                    filled = int(bar_len * counter / total_iter)
                    bar = '#' * filled + '-' * (bar_len - filled)
                    sys.stdout.write(f"\r    [{bar}] {pct:.0f}% ({counter}/{total_iter}) | {thr_label} | {r_name} | SL=${sl} | RR={rr}  ")
                    sys.stdout.flush()

    elapsed = time.time() - t_gs
    print(f"\n[OK] Grid search done in {elapsed:.1f}s")

    # 6. Report
    generate_report(results)

    # Print Top 10
    sorted_res = sorted(results, key=lambda x: x['final_equity'], reverse=True)
    print("\n[TOP 10 CONFIGURATIONS]")
    for r in sorted_res[:10]:
        wr = (r['wins'] / r['trades'] * 100) if r['trades'] > 0 else 0
        pnl = r['final_equity'] - 10000
        print(f" - {r['threshold_label']} | {r['ratio']} | SL ${r['sl']} | RR 1:{r['rr']}"
              f" -> ${r['final_equity']:.2f} (+{pnl/100:.1f}%) | WR: {wr:.1f}% | Trades: {r['trades']}")
