import requests
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import base64
import os
import sys
from io import BytesIO

# ==========================================
# --- CONFIGURATION ---
# ==========================================
SYMBOL = "ETHUSDT"
FILENAME = "eth_2025_data.csv"
# We want 2025 data. From today (Feb 2026) back to Jan 1 2025 is roughly 415 days.
# Let's fetch 450 to be safe.
DAYS_TO_FETCH = 450 

# Imbalance Ratios: max allowed ratio of opposing/dominant volume in wick
# Lower value = stricter imbalance required
# e.g. 0.20 means sell <= buy * 0.20, i.e. buy is 5x sell (very strong signal)
RATIOS = {
    '1.25x (80%)': 0.80,   # very loose: dominant barely above opposing
    '1.5x (67%)':  0.67,   # loose
    '2x (50%)':    0.50,   # moderate
    '2.5x (40%)':  0.40,   # moderate-strict
    '3x (33%)':    0.33,   # strict
    '4x (25%)':    0.25,   # very strict
    '5x (20%)':    0.20,   # extreme
    '6x (17%)':    0.17,   # ultra extreme
}

# Stop Loss candidates for ETH (Price ~2000-3500)
# 150 pips on BTC (~50k) is 0.3%.
# 0.3% on ETH (~2500) is ~7.5 pips.
SL_PARAMETERS = [5, 8, 12, 15] 

RISK_REWARD = 1.6

# ==========================================
# --- 1. DOWNLOADER ---
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
        print(f"[*] Downloading {symbol} ({days} days = ~{total_minutes:,} candles) from Binance Futures...")
        print(f"    This may take 2-5 minutes. Please wait.")
        t_start = time.time()
        
        while current_start < end_ts:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": "1m", 
                "startTime": current_start,
                "limit": 1500
            }
            
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=10)
                
                if r.status_code == 418 or r.status_code == 429:
                    print(f"\n[WARNING] Rate Limit Hit ({r.status_code}). Waiting 60s...")
                    time.sleep(60)
                    continue
                
                if r.status_code != 200:
                    print(f"\n[WARNING] API Error: {r.status_code}. Retrying...")
                    time.sleep(2)
                    continue
                
                data = r.json()
                if not data: break
                
                # Time, Open, High, Low, Close, Vol, ..., TakerBuyVol(9)
                for k in data:
                    all_klines.append([k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9])])
                
                last_ts = data[-1][0]
                current_start = last_ts + 60000
                
                # Progress every 10,000 candles
                if len(all_klines) % 10000 < 1500:
                    pct = min(len(all_klines) / total_minutes * 100, 100)
                    elapsed = time.time() - t_start
                    eta = (elapsed / max(pct, 0.1)) * (100 - pct)
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = '#' * filled + '-' * (bar_len - filled)
                    sys.stdout.write(f"\r    [{bar}] {pct:.0f}% | {len(all_klines):,} candles | ETA: {eta:.0f}s  ")
                    sys.stdout.flush()
                
                # Safe Delay
                time.sleep(0.2) 
                
            except Exception as e:
                print(f"\n[ERROR] Network error: {e}")
                time.sleep(2)
                continue
        
        elapsed = time.time() - t_start
        print(f"\n[OK] Download complete! Total candles: {len(all_klines):,} in {elapsed:.0f}s")
        df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'taker_buy_vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

# ==========================================
# --- 2. DATA PROCESSING ---
# ==========================================
def process_data(df_1m):
    print("[*] Resampling to 30m and calculating Wick Deltas...")
    
    # Calculate Buy/Sell Vol
    # Sell Vol = Total - Buy
    df_1m['sell_vol'] = df_1m['vol'] - df_1m['taker_buy_vol']
    
    # Resample to 30m
    df_1h = df_1m.set_index('ts').resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna().reset_index()
    
    # Wick Delta Calculation
    df_1m['h1_ts'] = df_1m['ts'].dt.floor('30min')
    merged = pd.merge(df_1m, df_1h[['ts', 'open', 'close']], left_on='h1_ts', right_on='ts', suffixes=('', '_h'))
    
    merged['wick_top'] = merged[['open_h', 'close_h']].max(axis=1)
    merged['wick_bot'] = merged[['open_h', 'close_h']].min(axis=1)
    
    # Upper Wick (Short Signal Area)
    mask_upper = merged['close'] >= merged['wick_top']
    upper_buys = merged[mask_upper].groupby('h1_ts')['taker_buy_vol'].sum()
    upper_sells = merged[mask_upper].groupby('h1_ts')['sell_vol'].sum()
    
    # Lower Wick (Long Signal Area)
    mask_lower = merged['close'] <= merged['wick_bot']
    lower_buys = merged[mask_lower].groupby('h1_ts')['taker_buy_vol'].sum()
    lower_sells = merged[mask_lower].groupby('h1_ts')['sell_vol'].sum()
    
    df_1h = df_1h.set_index('ts')
    df_1h['up_wick_buy'] = upper_buys
    df_1h['up_wick_sell'] = upper_sells
    df_1h['lo_wick_buy'] = lower_buys
    df_1h['lo_wick_sell'] = lower_sells
    df_1h.fillna(0, inplace=True)
    
    return df_1h.reset_index()

# ==========================================
# --- 3. BACKTEST ENGINE ---
# ==========================================
def run_backtest(df, threshold, ratio, sl_pips):
    equity = 10000
    equity_curve = [10000]
    trades_count = 0
    wins = 0
    trade_log = []
    
    in_pos = False
    pos_type = None
    entry = 0
    sl = 0
    tp = 0
    
    for i, row in df.iterrows():
        # Exit Logic
        if in_pos:
            hit_sl = False
            hit_tp = False
            
            if pos_type == 'Short':
                if row['high'] >= sl: hit_sl = True
                if row['low'] <= tp: hit_tp = True
            else:
                if row['low'] <= sl: hit_sl = True
                if row['high'] >= tp: hit_tp = True
            
            exit_flag = False
            pnl = 0
            
            if hit_sl: # Priority to SL if both hit (conservative) generally, but here checking both
                if hit_tp: # Both hit in same candle -> Assume Loss for safety (or check OHLC path if possible)
                     pnl = -abs(entry - sl)
                     exit_flag = True
                else: 
                    pnl = -abs(entry - sl)
                    exit_flag = True
            elif hit_tp:
                pnl = abs(entry - tp)
                exit_flag = True
                wins += 1
                
            if exit_flag:
                equity += pnl
                equity_curve.append(equity)
                trades_count += 1
                in_pos = False
                continue

        # Entry Logic
        if not in_pos:
            # Short Signal
            if (row['up_wick_buy'] >= threshold) and (row['up_wick_sell'] <= row['up_wick_buy'] * ratio):
                in_pos = True
                pos_type = 'Short'
                entry = row['close']
                sl = row['high'] + sl_pips
                risk = abs(sl - entry)
                tp = entry - (risk * RISK_REWARD)
            
            # Long Signal
            elif (row['lo_wick_sell'] >= threshold) and (row['lo_wick_buy'] <= row['lo_wick_sell'] * ratio):
                in_pos = True
                pos_type = 'Long'
                entry = row['close']
                sl = row['low'] - sl_pips
                risk = abs(entry - sl)
                tp = entry + (risk * RISK_REWARD)
                
    return trades_count, wins, equity_curve

# ==========================================
# --- 4. REPORT GENERATION ---
# ==========================================
def generate_report(results):
    # Sort by Equity
    sorted_res = sorted(results, key=lambda x: x['final_equity'], reverse=True)
    
    html = """<html><head><title>ETH 2025 Backtest</title>
    <style>
        body{font-family: sans-serif; padding: 20px;}
        table{border-collapse: collapse; width: 100%;}
        th, td{border: 1px solid #ddd; padding: 8px; text-align: left;}
        tr:nth-child(even){background-color: #f2f2f2;}
        th{background-color: #4CAF50; color: white;}
    </style></head><body>"""
    
    html += "<h1>ETHUSDTPERP Backtest Results (2025)</h1>"
    html += "<table><tr><th>Rank</th><th>Params (Thr / Ratio / SL)</th><th>Trades</th><th>WinRate</th><th>Final Equity</th></tr>"
    
    for i, res in enumerate(sorted_res[:20]): # Top 20
        wr = (res['wins'] / res['trades'] * 100) if res['trades'] > 0 else 0
        html += f"<tr><td>{i+1}</td><td>{res['params']}</td><td>{res['trades']}</td><td>{wr:.1f}%</td><td><b>{res['final_equity']:.2f}</b></td></tr>"
    
    html += "</table></body></html>"
    
    with open("eth_backtest_report.html", "w") as f:
        f.write(html)
    print("\n[DONE] Report generated: eth_backtest_report.html")

# ==========================================
# --- MAIN ---
# ==========================================
if __name__ == "__main__":
    # 1. Load/Download Data
    if os.path.exists(FILENAME):
        print(f"[*] Loading {FILENAME}...")
        df_1m = pd.read_csv(FILENAME)
        df_1m['ts'] = pd.to_datetime(df_1m['ts'])
    else:
        # API Key provided by user
        API_KEY = "7n8HrqdOnOc1RfpRsle2QEFwBUerJKnxOpW09yjaxU4eocUzwjpjJIEGtEJyKebs"
        client = BinanceKlines(api_key=API_KEY)
        df_1m = client.download_history(SYMBOL, DAYS_TO_FETCH)
        if not df_1m.empty:
            df_1m.to_csv(FILENAME, index=False)
    
    # 2. Filter 2025
    df_1m = df_1m[(df_1m['ts'] >= '2025-01-01') & (df_1m['ts'] <= '2025-12-31')]
    
    if df_1m.empty:
        print("[ERROR] No data for 2025!")
        exit()
        
    df_h1 = process_data(df_1m)
    
    # 3. Calculate Threshold Candidates (Top 1% to Top 10% of wick volume)
    # We look at the distribution of 'up_wick_buy' and 'lo_wick_sell' to find significant anomalies.
    # We exclude zeros (candles with no wick volume).
    up_vols = df_h1[df_h1['up_wick_buy'] > 10]['up_wick_buy']
    lo_vols = df_h1[df_h1['lo_wick_sell'] > 10]['lo_wick_sell']
    all_vols = pd.concat([up_vols, lo_vols])
    
    # Top 10% down to Top 1% (percentiles 90 to 99)
    percentile_levels = list(range(90, 100))  # 90,91,...,99
    thresholds = {}  # {int_val: str_label}
    for p in percentile_levels:
        val = int(all_vols.quantile(p / 100))
        label = f"Top {100 - p}%"
        if val not in thresholds:  # keep first occurrence (lowest percentile = most selective label)
            thresholds[val] = label
    
    print(f"[*] Volume Percentile Thresholds (ETH 2025):")
    for val in sorted(thresholds.keys()):
        print(f"    {thresholds[val]}: {val:,}")
    
    # 4. Grid Search
    results = []
    total_iter = len(thresholds) * len(RATIOS) * len(SL_PARAMETERS)
    print(f"\n[*] Starting Grid Search... ({total_iter} combinations)")
    counter = 0
    bar_len = 40
    
    for thr_val, thr_label in sorted(thresholds.items()):
        for r_name, r_val in RATIOS.items():
            for sl in SL_PARAMETERS:
                counter += 1
                count, wins, eq = run_backtest(df_h1, thr_val, r_val, sl)
                
                results.append({
                    'params': f"{thr_label} (>{thr_val:,}) | Imb {r_name} | SL {sl}",
                    'threshold_label': thr_label,
                    'threshold_val': thr_val,
                    'ratio': r_name,
                    'sl': sl,
                    'trades': count,
                    'wins': wins,
                    'final_equity': eq[-1]
                })
                
                # Progress bar
                pct = counter / total_iter * 100
                filled = int(bar_len * counter / total_iter)
                bar = '#' * filled + '-' * (bar_len - filled)
                sys.stdout.write(f"\r    [{bar}] {pct:.0f}% ({counter}/{total_iter}) | {thr_label} | Imb {r_name} | SL={sl}  ")
                sys.stdout.flush()
    
    print(f"\n[OK] Grid search complete!")
    
    # 5. Report
    generate_report(results)
    
    # Print Top 5
    sorted_res = sorted(results, key=lambda x: x['final_equity'], reverse=True)
    print("\n[TOP 5 CONFIGURATIONS]")
    for r in sorted_res[:5]:
        wr = (r['wins'] / r['trades'] * 100) if r['trades'] > 0 else 0
        print(f" - {r['params']} -> Equity: {r['final_equity']:.2f} | WR: {wr:.1f}% | Trades: {r['trades']}")

