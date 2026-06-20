"""
BTC 15M Scalp/Swing Backtest — Realistic Version
Uses shared backtest_engine.py
Fixes: look-ahead bias, slippage, 5% risk, Binance commissions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest_engine import (
    BinanceKlines, resample_to_tf, run_grid_search, generate_report, run_backtest,
    API_KEY
)
import pandas as pd

# ==========================================
# --- 15M CONFIGURATION ---
# ==========================================
SYMBOL   = "BTCUSDT"
FILENAME = "btc_2025_data.csv"
TF_MIN   = 15
REPORT   = "btc_15m_report.html"

# Expanded range of parameters
RATIOS = {
    '1.5x (67%)':  0.67,
    '2x (50%)':    0.50,
    '2.5x (40%)':  0.40,
    '3x (33%)':    0.33,
    '4x (25%)':    0.25,
    '5x (20%)':    0.20,
}

# Volume thresholds lower to capture more moves
VOLUME_THRESHOLDS = {
    '>1000': 1000,
    '>1500': 1500,
    '>2000': 2000,
    '>2500': 2500,
    '>3000': 3000,
    '>4000': 4000,
}

# SL Buffer (added to signal candle extremum)
SL_BUFFERS = [30, 50, 75, 100]
RR_VALUES  = [1.2, 1.6, 2.0, 2.5, 3.0]

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

    # 3. Resample to 15M
    df_tf = resample_to_tf(df_1m, TF_MIN)

    # 4. Show reference percentiles
    up_vols = df_tf[df_tf['up_wick_buy'] > 1]['up_wick_buy']
    lo_vols = df_tf[df_tf['lo_wick_sell'] > 1]['lo_wick_sell']
    all_vols = pd.concat([up_vols, lo_vols])
    print("\n[*] 15M Volume percentiles:")
    for p in [80, 85, 90, 95, 98, 99]:
        print(f"    Top {100-p}%: {all_vols.quantile(p/100):.1f}")

    # 5. Grid search
    results = run_grid_search(df_tf, VOLUME_THRESHOLDS, RATIOS, SL_BUFFERS, RR_VALUES)

    # 6. Detailed run for best result
    sorted_res = sorted(results, key=lambda x: x['final_equity'], reverse=True)
    best = sorted_res[0]
    print(f"\n[*] Best config: {best['threshold_label']} | Ratio {best['ratio']} | SL {best['sl']} | RR {best['rr']}")
    print("[*] Running detailed backtest for charts...")
    
    # Run single detailed pass
    _, _, _, detailed_log = run_backtest(
        df_tf, 
        best['threshold_val'], 
        RATIOS[best['ratio']], # Grid search stores label as ratio key? No, backtest engine stores 'ratio' as label in result dict.
                               # Wait, run_grid_search implementation: 
                               # results.append({ ..., 'ratio': r_name, ... })
                               # So best['ratio'] is the label (e.g. '2x (50%)').
                               # We need the value.
        best['sl'], 
        best['rr'], 
        detailed=True
    )

    # 7. Report
    generate_report(results, "15M Scalp", REPORT, detailed_trades=detailed_log)

    # 8. Print top 10
    print("\n[TOP 10 — 15M SCALP]")
    for r in sorted_res[:10]:
        wr = (r['wins'] / r['trades'] * 100) if r['trades'] > 0 else 0
        pnl_pct = (r['final_equity'] - 10000) / 100
        print(f"  {r['threshold_label']} | {r['ratio']} | SL ${r['sl']} | RR 1:{r['rr']}"
              f"  ->  ${r['final_equity']:.2f} (+{pnl_pct:.1f}%) | WR {wr:.1f}% | {r['trades']} trades")
