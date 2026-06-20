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
TIMEFRAME = "15min"
TF_LABEL = "15M"
TF_MINUTES = 15

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

def prepare_tf_data(df_1m):
    print(f"[INFO] Resampling 1m data to {TF_LABEL} and calculating wick volumes...")
    
    # Resample to target timeframe
    df_tf = df_1m.resample(TIMEFRAME).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Map 1m bars to their corresponding TF bar
    df_1m['tf_ts'] = df_1m.index.floor(TIMEFRAME)
    
    # Merge TF open/close back to 1m data to determine wick boundaries
    merged = pd.merge(df_1m.reset_index(), df_tf[['open', 'close']].reset_index(), 
                      left_on='tf_ts', right_on='ts', suffixes=('', '_tf'))
                      
    if 'ts_tf' in merged.columns:
        merged.drop(columns=['ts_tf'], inplace=True)
    
    merged['wick_top'] = merged[['open_tf', 'close_tf']].max(axis=1)
    merged['wick_bot'] = merged[['open_tf', 'close_tf']].min(axis=1)
    
    # Upper wick volumes (1m closes above the TF body)
    mask_upper = merged['close'] >= merged['wick_top']
    upper_buys = merged[mask_upper].groupby('tf_ts')['taker_buy_vol'].sum()
    upper_sells = merged[mask_upper].groupby('tf_ts')['sell_vol'].sum()
    
    # Lower wick volumes (1m closes below the TF body)
    mask_lower = merged['close'] <= merged['wick_bot']
    lower_buys = merged[mask_lower].groupby('tf_ts')['taker_buy_vol'].sum()
    lower_sells = merged[mask_lower].groupby('tf_ts')['sell_vol'].sum()
    
    # Assign to TF dataframe
    df_tf['up_wick_buy'] = upper_buys
    df_tf['up_wick_sell'] = upper_sells
    df_tf['lo_wick_buy'] = lower_buys
    df_tf['lo_wick_sell'] = lower_sells
    
    df_tf.fillna(0, inplace=True)
    
    # Calculate the total volume in each wick for percentiles
    df_tf['up_wick_total'] = df_tf['up_wick_buy'] + df_tf['up_wick_sell']
    df_tf['lo_wick_total'] = df_tf['lo_wick_buy'] + df_tf['lo_wick_sell']
    
    # Calculate global percentiles for the entire year to define "Anomaly"
    up_wick_totals_nonzero = df_tf[df_tf['up_wick_total'] > 0]['up_wick_total']
    lo_wick_totals_nonzero = df_tf[df_tf['lo_wick_total'] > 0]['lo_wick_total']
    
    # Combine all wick volumes to find global volume percentiles
    all_wick_volumes = pd.concat([up_wick_totals_nonzero, lo_wick_totals_nonzero])
    
    percentiles = {}
    for p in range(1, 11): # 1% to 10%
        # "Top X%" means the (100-X)th percentile
        percentiles[p] = np.percentile(all_wick_volumes, 100 - p)
        
    print(f"[INFO] Global Anomaly Thresholds Calculated ({TF_LABEL}):")
    for p, val in percentiles.items():
        print(f"  Top {p}% Volume Threshold: {val:.2f} BTC")
        
    return df_tf, percentiles

def backtest_strategy(df_tf_orig, df_1m, top_percent, imbalance_ratio, rr_ratio, sl_buffer_pct, percentiles_dict):
    volume_threshold = percentiles_dict[top_percent]
    
    # Work on a copy to avoid mutating original df across grid iterations
    df_tf = df_tf_orig.copy()
    
    # Vectorized signal generation for speed
    
    # SHORT SIGNAL (Anomalous Upper Wick)
    # 1. Total volume in upper wick >= threshold
    # 2. Buy volume >= Imbalance Ratio * Sell volume (e.g., 1.6 * Sell)
    short_condition = (df_tf['up_wick_total'] >= volume_threshold) & \
                      (df_tf['up_wick_buy'] >= df_tf['up_wick_sell'] * imbalance_ratio) & \
                      (df_tf['up_wick_sell'] > 0) # ensure no div by zero weirdness
                      
    # LONG SIGNAL (Anomalous Lower Wick)
    # 1. Total volume in lower wick >= threshold
    # 2. Sell volume >= Imbalance Ratio * Buy volume
    long_condition = (df_tf['lo_wick_total'] >= volume_threshold) & \
                     (df_tf['lo_wick_sell'] >= df_tf['lo_wick_buy'] * imbalance_ratio) & \
                     (df_tf['lo_wick_buy'] > 0)
                     
    df_tf['signal'] = 0
    df_tf.loc[short_condition, 'signal'] = -1
    df_tf.loc[long_condition, 'signal'] = 1
    
    # Skip candles with BOTH short and long signals (double signal = ambiguous)
    both_signals = short_condition & long_condition
    df_tf.loc[both_signals, 'signal'] = 0
    
    signals = df_tf[df_tf['signal'] != 0]
    
    if len(signals) == 0:
        return {'Total Trades': 0, 'Winrate %': 0.0, 'Net Profit ($)': 0.0, 'Max Drawdown (%)': 0.0, 'Profit Factor': 0.0, 'trades': [], 'equity_curve': [INITIAL_CAPITAL]}

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    
    # convert df_1m to arrays for extremely fast searching
    ts_1m = df_1m.index.values
    high_1m = df_1m['high'].values
    low_1m = df_1m['low'].values
    close_1m = df_1m['close'].values
    
    ts_tf = df_tf.index.values
    
    fee_rate_entry = 0.0004 # default taker
    fee_rate_exit = 0.0004 # hit SL = taker, hit TP = maker (but assuming taker for conservative)
    slip_rate = 0.0001
    
    current_idx_tf = 0
    while current_idx_tf < len(df_tf) - 1:
        # Check signal on current TF bar close
        signal = df_tf['signal'].iloc[current_idx_tf]
        
        if signal != 0:
            # Enter trade at close of TF candle
            bar = df_tf.iloc[current_idx_tf]
            entry_time = df_tf.index[current_idx_tf]
            # Time of the very next 1m bar opening (after TF candle closes)
            execution_time_approx = entry_time + pd.Timedelta(minutes=TF_MINUTES)
            
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
                    
                    # Advance the outer TF loop past this trade duration
                    exit_ts = ts_1m[m1_idx]
                    # Find next TF bar index after the exit
                    next_tf_idx = np.searchsorted(ts_tf, exit_ts)
                    current_idx_tf = max(current_idx_tf, next_tf_idx) # advance past exit candle
                    break
            
            if not trade_closed: # Hit end of data
                break 
                
        current_idx_tf += 1

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

def run_grid_search(df_tf, df_1m, percentiles_dict):
    top_percents = list(range(1, 11)) # 1 to 10%
    imbalance_ratios = [1.6, 2.0, 3.0, 4.0] # 60% (1.6x), 100% (2x), 200% (3x), 300% (4x)
    rr_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
    sl_buffers = [0.0005, 0.001, 0.0015] # 0.05%, 0.1%, 0.15% buffer behind high/low
    
    results = []
    
    total_iterations = len(top_percents) * len(imbalance_ratios) * len(rr_ratios) * len(sl_buffers)
    print(f"\n[INFO] Starting {TF_LABEL} Grid Search across {total_iterations} combinations...")
    
    i = 0
    for tp, imb, rr, sl_buf in product(top_percents, imbalance_ratios, rr_ratios, sl_buffers):
        i += 1
        res = backtest_strategy(df_tf, df_1m, tp, imb, rr, sl_buf, percentiles_dict)
        if res is not None:
            res['SL Buffer %'] = sl_buf * 100
            results.append(res)
        
        if i % 20 == 0:
            print(f"  Progress: {i}/{total_iterations}...")
            
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['trades', 'equity_curve']} for r in results])
    df_results.sort_values('Net Profit ($)', ascending=False, inplace=True)
    
    print(f"\n[OK] {TF_LABEL} Grid Search Complete ({len(df_results)} results). Top 15:")
    print(df_results.head(15).to_string(index=False))
    
    df_results.to_csv(f"anomaly_{TF_LABEL.lower()}_grid_results.csv", index=False)
    print(f"\n[INFO] Results saved to anomaly_{TF_LABEL.lower()}_grid_results.csv")
    
    best_res = max(results, key=lambda x: x['Net Profit ($)'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(best_res['equity_curve'])
    plt.title(f"Best {TF_LABEL}: Top {best_res['Top % Volume']}% Vol, Imb {best_res['Imbalance Ratio']}x, R:R {best_res['R:R']}, SL Buf {best_res.get('SL Buffer %', '?')}%")
    plt.ylabel("Capital ($)")
    plt.grid(True)
    plt.savefig(f"best_{TF_LABEL.lower()}_anomaly_equity.png")
    print(f"\n[INFO] Best equity curve saved to best_{TF_LABEL.lower()}_anomaly_equity.png")

if __name__ == "__main__":
    df_1m = load_data()
    if df_1m is not None:
        df_tf, percentiles = prepare_tf_data(df_1m)
        run_grid_search(df_tf, df_1m, percentiles)
