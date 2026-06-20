"""
BTC 5M Scalp Backtest — Realistic Version
Uses shared backtest_engine.py
Fixes: look-ahead bias, slippage, 5% risk, Binance commissions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest_engine import (
    BinanceKlines, resample_to_tf, run_grid_search, generate_report,
    API_KEY
)
import pandas as pd

# ==========================================
# --- 5M CONFIGURATION ---
# ==========================================
SYMBOL   = "BTCUSDT"
FILENAME = "btc_2025_data.csv"
TF_MIN   = 5
REPORT   = "btc_5m_report.html"

RATIOS = {
    '1.25x (80%)': 0.80,
    '1.5x (67%)':  0.67,
    '2x (50%)':    0.50,
    '2.5x (40%)':  0.40,
    '3x (33%)':    0.33,
    '4x (25%)':    0.25,
}

# Thresholds tuned for 5M (higher volume candles)
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
RR_VALUES     = [1.2, 1.6, 2.0, 2.5]

# ==========================================
# --- MAIN ---
# ==========================================
if __name__ == "__main__":
    # 1. Load data
    client = BinanceKlines(api_key=API_KEY)
    df_1m = client.download_history(SYMBOL, 450, FILENAME)

    # 2. Filter 2025
    df_1m = df_1m[(df_1m['ts'] >= '2025-01-01') & (df_1m['ts'] < '2026-01-01')]
    print(f"[*] 2025 data: {len(df_1m):,} 1M candles")

    # 3. Resample to 5M
    df_tf = resample_to_tf(df_1m, TF_MIN)

    # 4. Show reference percentiles
    up_vols = df_tf[df_tf['up_wick_buy'] > 1]['up_wick_buy']
    lo_vols = df_tf[df_tf['lo_wick_sell'] > 1]['lo_wick_sell']
    all_vols = pd.concat([up_vols, lo_vols])
    print("\n[*] 5M Volume percentiles:")
    for p in [90, 93, 95, 97, 98, 99]:
        print(f"    Top {100-p}%: {all_vols.quantile(p/100):.1f}")

    # 5. Grid search
    results = run_grid_search(df_tf, VOLUME_THRESHOLDS, RATIOS, SL_PARAMETERS, RR_VALUES)

    # 6. Report
    generate_report(results, "5M Scalp", REPORT)

    # 7. Print top 10
    sorted_res = sorted(results, key=lambda x: x['final_equity'], reverse=True)
    print("\n[TOP 10 — 5M SCALP]")
    for r in sorted_res[:10]:
        wr = (r['wins'] / r['trades'] * 100) if r['trades'] > 0 else 0
        pnl_pct = (r['final_equity'] - 10000) / 100
        print(f"  {r['threshold_label']} | {r['ratio']} | SL ${r['sl']} | RR 1:{r['rr']}"
              f"  ->  ${r['final_equity']:.2f} (+{pnl_pct:.1f}%) | WR {wr:.1f}% | {r['trades']} trades")
