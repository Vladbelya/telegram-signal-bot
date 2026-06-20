import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from datetime import time

SYMBOL = "BTCUSDT"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "btc_2025_data.csv")
INITIAL_CAPITAL = 10000.0

def load_data(filepath=DATA_FILE):
    if not os.path.exists(filepath):
        print(f"[ERROR] Data file {filepath} not found. Please ensure 1m data is available.")
        return None
        
    print(f"[INFO] Loading 1m data from {filepath}...")
    df_1m = pd.read_csv(filepath)
    time_col = 'ts' if 'ts' in df_1m.columns else ('timestamp' if 'timestamp' in df_1m.columns else df_1m.columns[0])
    df_1m[time_col] = pd.to_datetime(df_1m[time_col])
    
    # Filter for 2025
    df_1m = df_1m[(df_1m[time_col] >= '2025-01-01') & (df_1m[time_col] <= '2025-12-31')]
    
    # Calculate Buy and Sell Volumes for each minute
    if 'sell_vol' not in df_1m.columns:
        df_1m['sell_vol'] = df_1m['vol'] - df_1m['taker_buy_vol']
    
    df_1m.set_index(time_col, inplace=True)
    df_1m.sort_index(inplace=True)
    return df_1m

def prepare_5m_data(df_1m):
    print("[INFO] Resampling 1m data to 5m and calculating wick volumes...")
    
    # Resample to 5m
    df_5m = df_1m.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Map 1m bars to their corresponding 5m bar
    df_1m['m5_ts'] = df_1m.index.floor('5min')
    
    # Merge 5m open/close back to 1m data to determine wick boundaries
    merged = pd.merge(df_1m.reset_index(), df_5m[['open', 'close']].reset_index(), 
                      left_on='m5_ts', right_on='ts', suffixes=('', '_m5'))
                      
    if 'ts_m5' in merged.columns:
        merged.drop(columns=['ts_m5'], inplace=True)
    
    merged['wick_top'] = merged[['open_m5', 'close_m5']].max(axis=1)
    merged['wick_bot'] = merged[['open_m5', 'close_m5']].min(axis=1)
    
    # Upper wick volumes (1m closes above the 5m body)
    mask_upper = merged['close'] >= merged['wick_top']
    upper_buys = merged[mask_upper].groupby('m5_ts')['taker_buy_vol'].sum()
    upper_sells = merged[mask_upper].groupby('m5_ts')['sell_vol'].sum()
    
    # Lower wick volumes (1m closes below the 5m body)
    mask_lower = merged['close'] <= merged['wick_bot']
    lower_buys = merged[mask_lower].groupby('m5_ts')['taker_buy_vol'].sum()
    lower_sells = merged[mask_lower].groupby('m5_ts')['sell_vol'].sum()
    
    # Assign to 5m dataframe
    df_5m['up_wick_buy'] = upper_buys
    df_5m['up_wick_sell'] = upper_sells
    df_5m['lo_wick_buy'] = lower_buys
    df_5m['lo_wick_sell'] = lower_sells
    
    df_5m.fillna(0, inplace=True)
    
    # Calculate the total volume in each wick for percentiles
    df_5m['up_wick_total'] = df_5m['up_wick_buy'] + df_5m['up_wick_sell']
    df_5m['lo_wick_total'] = df_5m['lo_wick_buy'] + df_5m['lo_wick_sell']
    
    # Calculate global percentiles for the entire year to define "Anomaly"
    up_wick_totals_nonzero = df_5m[df_5m['up_wick_total'] > 0]['up_wick_total']
    lo_wick_totals_nonzero = df_5m[df_5m['lo_wick_total'] > 0]['lo_wick_total']
    
    # Combine all wick volumes to find global volume percentiles
    all_wick_volumes = pd.concat([up_wick_totals_nonzero, lo_wick_totals_nonzero])
    
    percentiles = {}
    for p in range(1, 11): # 1% to 10%
        # "Top X%" means the (100-X)th percentile
        percentiles[p] = np.percentile(all_wick_volumes, 100 - p)
        
    print("[INFO] Global Anomaly Thresholds Calculated:")
    for p, val in percentiles.items():
        print(f"  Top {p}% Volume Threshold: {val:.2f} BTC")
        
    return df_5m, percentiles

def backtest_strategy(df_5m_orig, df_1m, top_percent, imbalance_ratio, rr_ratio, sl_buffer_pct, percentiles_dict):
    volume_threshold = percentiles_dict[top_percent]
    
    # Work on a copy to avoid mutating original df across grid iterations
    df_5m = df_5m_orig.copy()
    
    # Vectorized signal generation for speed
    
    # SHORT SIGNAL (Anomalous Upper Wick)
    # 1. Total volume in upper wick >= threshold
    # 2. Buy volume >= Imbalance Ratio * Sell volume (e.g., 1.6 * Sell)
    short_condition = (df_5m['up_wick_total'] >= volume_threshold) & \
                      (df_5m['up_wick_buy'] >= df_5m['up_wick_sell'] * imbalance_ratio) & \
                      (df_5m['up_wick_sell'] > 0) # ensure no div by zero weirdness
                      
    # LONG SIGNAL (Anomalous Lower Wick)
    # 1. Total volume in lower wick >= threshold
    # 2. Sell volume >= Imbalance Ratio * Buy volume
    long_condition = (df_5m['lo_wick_total'] >= volume_threshold) & \
                     (df_5m['lo_wick_sell'] >= df_5m['lo_wick_buy'] * imbalance_ratio) & \
                     (df_5m['lo_wick_buy'] > 0)
                     
    df_5m['signal'] = 0
    df_5m.loc[short_condition, 'signal'] = -1
    df_5m.loc[long_condition, 'signal'] = 1
    
    # Skip candles with BOTH short and long signals (double signal = ambiguous)
    both_signals = short_condition & long_condition
    df_5m.loc[both_signals, 'signal'] = 0
    
    signals = df_5m[df_5m['signal'] != 0]
    
    if len(signals) == 0:
        return {'Total Trades': 0, 'Winrate %': 0.0, 'Net Profit ($)': 0.0, 'Max Drawdown (%)': 0.0, 'Profit Factor': 0.0, 'trades': [], 'equity_curve': [INITIAL_CAPITAL]}

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    
    # Ensure 1m index is sorted for fast slicing
    time_col_1m = df_1m.index.name
    
    # Simulation loop
    in_pos = False
    pos_type = 0
    entry_price = 0
    sl_price = 0
    tp_price = 0
    size = 0
    trade_start_idx = None
    
    # convert df_1m to arrays for extremely fast searching
    ts_1m = df_1m.index.values
    high_1m = df_1m['high'].values
    low_1m = df_1m['low'].values
    close_1m = df_1m['close'].values
    
    ts_5m = df_5m.index.values
    
    fee_rate_entry = 0.0004 # default taker
    fee_rate_exit = 0.0004 # hit SL = taker, hit TP = maker (but assuming taker for conservative)
    slip_rate = 0.0001
    
    current_idx_5m = 0
    while current_idx_5m < len(df_5m) - 1:
        # Check signal on current 5m bar close
        signal = df_5m['signal'].iloc[current_idx_5m]
        
        if signal != 0:
            # Enter trade at close of 5m candle
            bar = df_5m.iloc[current_idx_5m]
            entry_time = df_5m.index[current_idx_5m]
            # Time of the very next 1m bar opening
            execution_time_approx = entry_time + pd.Timedelta(minutes=5)
            
            entry_price = bar['close']
            
            if signal == -1: # SHORT
                pos_type = -1
                sl_price = bar['high'] * (1 + sl_buffer_pct) # SL behind High + buffer
                # Prevent zero risk division if open=high=close
                dist_to_sl = abs(sl_price - entry_price)
                if dist_to_sl < 1: 
                    sl_price = entry_price + (entry_price * 0.001) # fallback 0.1% sl
                    dist_to_sl = abs(sl_price - entry_price)
                    
                risk_amt = capital * 0.01 # 1% of equity
                size = risk_amt / dist_to_sl
                tp_price = entry_price - (dist_to_sl * rr_ratio)
                
            elif signal == 1: # LONG
                pos_type = 1
                sl_price = bar['low'] * (1 - sl_buffer_pct) # SL behind Low - buffer
                dist_to_sl = abs(entry_price - sl_price)
                if dist_to_sl < 1:
                    sl_price = entry_price - (entry_price * 0.001)
                    dist_to_sl = abs(entry_price - sl_price)
                    
                risk_amt = capital * 0.01
                size = risk_amt / dist_to_sl
                tp_price = entry_price + (dist_to_sl * rr_ratio)
                
            # Simulate through 1m data starting from execution_time_approx
            start_m1_idx = np.searchsorted(ts_1m, np.datetime64(execution_time_approx))
            
            trade_closed = False
            for m1_idx in range(start_m1_idx, len(ts_1m)):
                h = high_1m[m1_idx]
                l = low_1m[m1_idx]
                c = close_1m[m1_idx]
                
                hit_sl = False
                hit_tp = False
                
                if pos_type == -1: # Short
                    if h >= sl_price: hit_sl = True
                    if l <= tp_price: hit_tp = True
                elif pos_type == 1: # Long
                    if l <= sl_price: hit_sl = True
                    if h >= tp_price: hit_tp = True
                    
                if hit_sl and hit_tp:
                    exit_price = sl_price # Conservative
                    exit_type = "SL (Conflict)"
                    trade_closed = True
                elif hit_sl:
                    exit_price = sl_price
                    exit_type = "SL"
                    trade_closed = True
                elif hit_tp:
                    exit_price = tp_price
                    exit_type = "TP"
                    trade_closed = True
                    
                if trade_closed:
                    # Calculate PnL strictly
                    if pos_type == 1: # Long
                        gross = (exit_price - entry_price) * size
                    else:
                        gross = (entry_price - exit_price) * size
                        
                    comm = (entry_price * size * fee_rate_entry) + (exit_price * size * fee_rate_exit)
                    slip_cost = (entry_price * size * slip_rate) + (exit_price * size * slip_rate)
                    net_pnl = gross - comm - slip_cost
                    
                    capital += net_pnl
                    equity_curve.append(capital)
                    
                    trades.append({
                        'Entry Time': np.datetime64(execution_time_approx),
                        'Exit Time': ts_1m[m1_idx],
                        'Type': 'Long' if pos_type == 1 else 'Short',
                        'Size (BTC)': size,
                        'Entry': entry_price,
                        'SL': sl_price,
                        'TP': tp_price,
                        'Exit': exit_price,
                        'Net PnL': net_pnl,
                        'Result': 'Win' if net_pnl > 0 else 'Loss',
                        'Exit Type': exit_type
                    })
                    
                    # Advance the outer 5m loop past this trade duration
                    exit_ts = ts_1m[m1_idx]
                    # Find next 5m bar index after the exit
                    next_5m_idx = np.searchsorted(ts_5m, exit_ts)
                    current_idx_5m = max(current_idx_5m, next_5m_idx) # advance past exit candle
                    break
            
            if not trade_closed: # Hit end of data
                break 
                
        current_idx_5m += 1

    # Calculate metrics
    total_trades = len(trades)
    if total_trades == 0:
        return {'Total Trades': 0, 'Winrate %': 0.0, 'Net Profit ($)': 0.0, 'Max Drawdown (%)': 0.0, 'Profit Factor': 0.0, 'trades': trades, 'equity_curve': equity_curve}
        
    wins = sum(1 for t in trades if t['Result'] == 'Win')
    winrate = (wins / total_trades) * 100
    net_profit = capital - INITIAL_CAPITAL
    
    eq_series = pd.Series(equity_curve)
    max_dd = ((eq_series.cummax() - eq_series) / eq_series.cummax()).max() * 100
    
    gross_profits = sum(t['Net PnL'] for t in trades if t['Net PnL'] > 0)
    gross_losses = sum(t['Net PnL'] for t in trades if t['Net PnL'] < 0)
    pf = gross_profits / abs(gross_losses) if gross_losses != 0 else np.inf
    
    return {
        'Top % Volume': top_percent,
        'Imbalance Ratio': imbalance_ratio,
        'R:R': rr_ratio,
        'Total Trades': total_trades,
        'Winrate %': round(winrate, 2),
        'Net Profit ($)': round(net_profit, 2),
        'Max Drawdown (%)': round(max_dd, 2),
        'Profit Factor': round(pf, 2),
        'trades': trades,
        'equity_curve': equity_curve
    }

def run_grid_search(df_5m, df_1m, percentiles_dict):
    top_percents = list(range(1, 11)) # 1 to 10%
    imbalance_ratios = [1.6, 2.0, 3.0, 4.0] # 60% (1.6x), 100% (2x), 200% (3x), 300% (4x)
    rr_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
    sl_buffers = [0.0005, 0.001, 0.0015] # 0.05%, 0.1%, 0.15% buffer behind high/low
    
    results = []
    
    total_iterations = len(top_percents) * len(imbalance_ratios) * len(rr_ratios) * len(sl_buffers)
    print(f"\n[INFO] Starting Grid Search across {total_iterations} combinations...")
    
    i = 0
    for tp, imb, rr, sl_buf in product(top_percents, imbalance_ratios, rr_ratios, sl_buffers):
        i += 1
        res = backtest_strategy(df_5m, df_1m, tp, imb, rr, sl_buf, percentiles_dict)
        if res is not None:
            res['SL Buffer %'] = sl_buf * 100
            results.append(res)
        
        if i % 20 == 0:
            print(f"  Progress: {i}/{total_iterations}...")
            
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['trades', 'equity_curve']} for r in results])
    df_results.sort_values('Net Profit ($)', ascending=False, inplace=True)
    
    print(f"\n[OK] Grid Search Complete ({len(df_results)} results). Top 15:")
    print(df_results.head(15).to_string(index=False))
    
    df_results.to_csv("anomaly_5m_grid_results.csv", index=False)
    print("\n[INFO] Results saved to anomaly_5m_grid_results.csv")
    
    best_res = max(results, key=lambda x: x['Net Profit ($)'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(best_res['equity_curve'])
    plt.title(f"Best: Top {best_res['Top % Volume']}% Vol, Imb {best_res['Imbalance Ratio']}x, R:R {best_res['R:R']}, SL Buf {best_res.get('SL Buffer %', '?')}%")
    plt.ylabel("Capital ($)")
    plt.grid(True)
    plt.savefig("best_5m_anomaly_equity.png")
    print("\n[INFO] Best equity curve saved to best_5m_anomaly_equity.png")

if __name__ == "__main__":
    df_1m = load_data()
    if df_1m is not None:
        df_5m, percentiles = prepare_5m_data(df_1m)
        run_grid_search(df_5m, df_1m, percentiles)
