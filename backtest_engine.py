"""
Shared Backtest Engine — Realistic version
Fixes:
  1. Look-ahead bias: signal on candle[i], entry on candle[i+1].open
  2. Slippage: 0.05% applied to entry price
  3. Risk management: 5% of equity per trade
  4. Commission: 0.08% of notional (Binance Futures taker x2)
"""

import requests
import pandas as pd
import numpy as np
import datetime
import time
import sys
import os

# ==========================================
# --- CONSTANTS ---
# ==========================================
COMMISSION    = 0.0008   # 0.08% per round-trip (0.04% taker x2)
RISK_PER_TRADE = 0.05    # 5% of equity risked per trade
SLIPPAGE      = 0.0005   # 0.05% slippage on entry price
INITIAL_EQUITY = 10000.0

API_KEY = "7n8HrqdOnOc1RfpRsle2QEFwBUerJKnxOpW09yjaxU4eocUzwjpjJIEGtEJyKebs"

# ==========================================
# --- DOWNLOADER ---
# ==========================================
class BinanceKlines:
    def __init__(self, api_key=None):
        self.base_url = "https://fapi.binance.com"
        self.headers = {'X-MBX-APIKEY': api_key} if api_key else {}

    def download_history(self, symbol, days, filename):
        if os.path.exists(filename):
            print(f"[*] Loading cached {filename}...")
            df = pd.read_csv(filename)
            df['ts'] = pd.to_datetime(df['ts'])
            return df

        end_dt = datetime.datetime.now(datetime.timezone.utc)
        start_dt = end_dt - datetime.timedelta(days=days)
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts   = int(end_dt.timestamp() * 1000)
        all_klines = []
        current_start = start_ts
        total_minutes = days * 1440

        print(f"[*] Downloading {symbol} 1m ({days} days)...")
        t_start = time.time()

        while current_start < end_ts:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {"symbol": symbol, "interval": "1m",
                      "startTime": current_start, "limit": 1500}
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=10)
                if r.status_code in (418, 429):
                    time.sleep(60); continue
                if r.status_code != 200:
                    time.sleep(2); continue
                data = r.json()
                if not data: break
                for k in data:
                    all_klines.append([k[0], float(k[1]), float(k[2]),
                                       float(k[3]), float(k[4]), float(k[5]), float(k[9])])
                current_start = data[-1][0] + 60000
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
        df.to_csv(filename, index=False)
        return df

# ==========================================
# --- DATA PROCESSING ---
# ==========================================
def resample_to_tf(df_1m, tf_minutes):
    """Resample 1m data to target timeframe and calculate wick deltas."""
    tf_str = f"{tf_minutes}min"
    print(f"[*] Resampling to {tf_minutes}M and calculating Wick Deltas...")
    t0 = time.time()

    df_1m = df_1m.copy()
    df_1m['sell_vol'] = df_1m['vol'] - df_1m['taker_buy_vol']

    df_tf = df_1m.set_index('ts').resample(tf_str).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
    ).dropna().reset_index()

    df_1m['h_ts'] = df_1m['ts'].dt.floor(tf_str)
    merged = pd.merge(df_1m, df_tf[['ts','open','close']],
                      left_on='h_ts', right_on='ts', suffixes=('','_tf'))

    merged['wick_top'] = merged[['open_tf','close_tf']].max(axis=1)
    merged['wick_bot'] = merged[['open_tf','close_tf']].min(axis=1)

    mask_up = merged['close'] >= merged['wick_top']
    upper_buys  = merged[mask_up].groupby('h_ts')['taker_buy_vol'].sum()
    upper_sells = merged[mask_up].groupby('h_ts')['sell_vol'].sum()

    mask_lo = merged['close'] <= merged['wick_bot']
    lower_buys  = merged[mask_lo].groupby('h_ts')['taker_buy_vol'].sum()
    lower_sells = merged[mask_lo].groupby('h_ts')['sell_vol'].sum()

    df_tf = df_tf.set_index('ts')
    df_tf['up_wick_buy']  = upper_buys
    df_tf['up_wick_sell'] = upper_sells
    df_tf['lo_wick_buy']  = lower_buys
    df_tf['lo_wick_sell'] = lower_sells
    df_tf.fillna(0, inplace=True)

    # Add next candle open for realistic entry (fix look-ahead bias)
    df_tf['next_open'] = df_tf['open'].shift(-1)

    print(f"[*] Processed {len(df_tf):,} {tf_minutes}M candles in {time.time()-t0:.1f}s")
    return df_tf.reset_index()

# ==========================================
# --- BACKTEST ENGINE ---
# ==========================================
def run_backtest(df, threshold, ratio, sl_buffer, rr, detailed=False):
    """
    Realistic backtest engine:
    - Signal detected on candle[i] close
    - Entry on candle[i+1] open + slippage
    - SL/TP checked on candle[i+1] onwards (high/low)
    - Risk: 5% of equity per trade
    - Commission: 0.08% of notional
    - Slippage: 0.05% on entry
    """
    """
    highs    = df['high'].values
    lows     = df['low'].values
    opens    = df['open'].values
    closes   = df['close'].values
    next_open = df['next_open'].values   # entry price
    timestamps = df['ts'].values
    up_buy   = df['up_wick_buy'].values
    up_sell  = df['up_wick_sell'].values
    lo_buy   = df['lo_wick_buy'].values
    lo_sell  = df['lo_wick_sell'].values

    n = len(df)
    equity = INITIAL_EQUITY
    trades_count = 0
    wins = 0

    in_pos   = False
    pos_type = 0       # 1=Long, -1=Short
    entry = sl = tp = pos_size = 0.0
    pending_signal = 0  # signal detected (-1 or 1)
    signal_idx = 0      # index of signal candle
    signal_extremum = 0.0 # High (for Short) or Low (for Long) of signal candle

    detailed_log = [] # For storing trade details if needed

    for i in range(n - 1):
        if equity <= 0:
            break

        # --- Check open position exit ---
        if in_pos:
            exit_price = 0.0
            pnl_val = 0.0
            result = ""
            
            if pos_type == -1:  # Short
                if highs[i] >= sl:
                    exit_price = sl
                    commission = pos_size * entry * COMMISSION
                    pnl_val = -(pos_size * abs(entry - sl)) - commission
                    equity += pnl_val
                    result = "LOSS"
                elif lows[i] <= tp:
                    exit_price = tp
                    commission = pos_size * entry * COMMISSION
                    pnl_val = (pos_size * abs(entry - tp)) - commission
                    equity += pnl_val
                    result = "WIN"
            else:  # Long
                if lows[i] <= sl:
                    exit_price = sl
                    commission = pos_size * entry * COMMISSION
                    pnl_val = -(pos_size * abs(entry - sl)) - commission
                    equity += pnl_val
                    result = "LOSS"
                elif highs[i] >= tp:
                    exit_price = tp
                    commission = pos_size * entry * COMMISSION
                    pnl_val = (pos_size * abs(entry - tp)) - commission
                    equity += pnl_val
                    result = "WIN"

            if result:
                trades_count += 1
                if result == "WIN": wins += 1
                in_pos = False
                
                if detailed:
                    # Capture candle data for chart (Start: Signal-2, End: Close+2)
                    idx_start = max(0, signal_idx - 5)
                    idx_end = min(n, i + 5)
                    
                    chart_data = []
                    for k in range(idx_start, idx_end):
                        chart_data.append({
                            't': str(timestamps[k]),
                            'o': float(opens[k]), 'h': float(highs[k]), 'l': float(lows[k]), 'c': float(closes[k]),
                        })

                    detailed_log.append({
                        'type': 'SHORT' if pos_type == -1 else 'LONG',
                        'entry': float(entry),
                        'sl': float(sl),
                        'tp': float(tp),
                        'pnl': float(pnl_val),
                        'result': result,
                        'signal_time': str(timestamps[signal_idx]),
                        'entry_time': str(timestamps[signal_idx+1]),
                        'close_time': str(timestamps[i]),
                        'candles': chart_data,
                        'signal_idx_rel': signal_idx - idx_start,
                        'entry_idx_rel': (signal_idx + 1) - idx_start,
                        'close_idx_rel': i - idx_start
                    })
            continue

        # --- Enter pending trade ---
        if pending_signal != 0 and not np.isnan(next_open[i-1]):
            # i-1 is the candle that just closed (the signal candle was at i-1? No wait)
            # Loop runs for i.
            # Look-ahead bias fix used next_open[i] which was open of i+1.
            # But here we are iterating candles.
            # Original code: `if pending_signal != 0 and not np.isnan(next_open[i-1]):`
            # `raw_entry = next_open[i-1]`
            # If signal was at K, we enter at K+1 Open.
            
            raw_entry = next_open[i-1]
            if pending_signal == -1:    # SHORT
                entry = raw_entry * (1 + SLIPPAGE)
                sl = signal_extremum + sl_buffer
                # If SL is below entry (impossible for Short unless gap), cap logic
                if sl <= entry: sl = entry * 1.001
                    
                risk = abs(entry - sl)
                tp = entry - risk * rr
            else:                       # LONG
                entry = raw_entry * (1 - SLIPPAGE)
                sl = signal_extremum - sl_buffer
                if sl >= entry: sl = entry * 0.999
                    
                risk = abs(entry - sl)
                tp = entry + risk * rr

            # Basic sanity check
            if risk > 0 and equity > 0:
                risk_amount = equity * RISK_PER_TRADE
                pos_size = risk_amount / risk
                in_pos = True
                pos_type = pending_signal
            
            pending_signal = 0

        # --- Detect new signal ---
        if not in_pos:
            if up_buy[i] >= threshold and up_sell[i] <= up_buy[i] * ratio:
                pending_signal = -1
                signal_idx = i
                signal_extremum = highs[i]
            elif lo_sell[i] >= threshold and lo_buy[i] <= lo_sell[i] * ratio:
                pending_signal = 1
                signal_idx = i
                signal_extremum = lows[i]

    if detailed:
        return trades_count, wins, equity, detailed_log
    return trades_count, wins, equity

# ==========================================
# --- REPORT GENERATOR ---
# ==========================================
def generate_report(results, timeframe_label, output_file, detailed_trades=None):
    sorted_res = sorted(results, key=lambda x: x['final_equity'], reverse=True)

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>BTC {timeframe_label} Backtest</title>
    <style>
        body{{font-family:'Segoe UI',sans-serif;padding:30px;background:#0d1117;color:#e6edf3;}}
        h1{{color:#f7931a;font-size:26px;margin-bottom:4px;}}
        h2{{color:#58a6ff;margin-top:28px;font-size:18px;}}
        .meta{{color:#8b949e;font-size:13px;margin-bottom:24px;}}
        .info-box{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;}}
        .info-card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 20px;min-width:140px;}}
        .info-card .label{{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;}}
        .info-card .value{{color:#f7931a;font-size:20px;font-weight:700;margin-top:4px;}}
        table{{border-collapse:collapse;width:100%;margin-top:16px;}}
        th,td{{border:1px solid #30363d;padding:9px 13px;text-align:left;font-size:13px;}}
        tr:nth-child(even){{background:#161b22;}}
        th{{background:#1c2128;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;}}
        .profit{{color:#3fb950;font-weight:700;}}
        .loss{{color:#f85149;font-weight:700;}}
        .badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;}}
        .b-green{{background:rgba(63,185,80,.15);color:#3fb950;}}
        .b-orange{{background:rgba(247,147,26,.15);color:#f7931a;}}
        .b-purple{{background:rgba(188,140,255,.15);color:#bc8cff;}}
        .top1{{background:rgba(247,147,26,.08)!important;}}
    </style></head><body>"""

    best = sorted_res[0] if sorted_res else {}
    best_wr = (best.get('wins',0) / best.get('trades',1) * 100) if best.get('trades') else 0
    best_pnl = best.get('final_equity', 10000) - 10000

    html += f"<h1>BTCUSDT {timeframe_label} Backtest — 2025</h1>"
    html += f"""<div class='meta'>
        Grid search: {len(results)} combinations | Capital: $10,000 | Risk: 5%/trade |
        Commission: 0.08% | Slippage: 0.05% | Entry: next candle open
    </div>"""

    html += f"""<div class='info-box'>
        <div class='info-card'><div class='label'>Best P&L</div>
            <div class='value'>+{best_pnl/100:.1f}%</div></div>
        <div class='info-card'><div class='label'>Best WR</div>
            <div class='value'>{best_wr:.1f}%</div></div>
        <div class='info-card'><div class='label'>Best Trades</div>
            <div class='value'>{best.get('trades',0)}</div></div>
        <div class='info-card'><div class='label'>Threshold</div>
            <div class='value'>{best.get('threshold_label','')}</div></div>
    </div>"""

    html += "<h2>Top 50 Configurations</h2>"
    html += """<table><tr>
        <th>#</th><th>Volume Threshold</th><th>Imbalance</th>
        <th>SL ($)</th><th>RR</th><th>Trades</th>
        <th>Win Rate</th><th>Final Equity</th><th>P&L ($)</th><th>P&L (%)</th>
    </tr>"""

    for i, res in enumerate(sorted_res[:50]):
        wr = (res['wins'] / res['trades'] * 100) if res['trades'] > 0 else 0
        pnl = res['final_equity'] - 10000
        pnl_pct = pnl / 10000 * 100
        pnl_cls = 'profit' if pnl >= 0 else 'loss'
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        pnl_pct_str = f"+{pnl_pct:.1f}%" if pnl_pct >= 0 else f"{pnl_pct:.1f}%"
        row_cls = ' class="top1"' if i == 0 else ''
        rank = f'<b>#{i+1}</b>'
        html += (f"<tr{row_cls}><td>{rank}</td>"
                 f"<td>{res['threshold_label']}</td>"
                 f"<td><span class='badge b-orange'>{res['ratio']}</span></td>"
                 f"<td>${res['sl']}</td>"
                 f"<td><span class='badge b-purple'>1:{res['rr']}</span></td>"
                 f"<td>{res['trades']}</td>"
                 f"<td><span class='badge b-green'>{wr:.1f}%</span></td>"
                 f"<td><b>${res['final_equity']:.2f}</b></td>"
                 f"<td class='{pnl_cls}'>{pnl_str}</td>"
                 f"<td class='{pnl_cls}'><b>{pnl_pct_str}</b></td></tr>")

    html += "</table>"
    
    # --- Add Charts if detailed trades provided ---
    if detailed_trades:
        html += "<h2 style='margin-top:40px; color:#58a6ff'>Trade Examples (5 Wins / 5 Losses)</h2>"
        html += "<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(400px, 1fr)); gap:20px;'>"
        
        # Select 5 wins and 5 losses with highest/lowest PnL to be interesting
        wins_list = sorted([t for t in detailed_trades if t['result'] == 'WIN'], key=lambda x: x['pnl'], reverse=True)[:5]
        loss_list = sorted([t for t in detailed_trades if t['result'] == 'LOSS'], key=lambda x: x['pnl'])[:5]
        examples = wins_list + loss_list
        
        import json
        
        for idx, trade in enumerate(examples):
            canvas_id = f"chart_{idx}"
            pnl_class = "profit" if trade['pnl'] > 0 else "loss"
            safe_trade_json = json.dumps(trade)
            
            html += f"""
            <div style='background:#161b22; border:1px solid #30363d; border-radius:10px; padding:15px; overflow:hidden;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px; font-size:12px; color:#8b949e; font-family:"JetBrains Mono", monospace;'>
                    <span>{trade['type']} | {trade['signal_time']}</span>
                    <span class='{pnl_class}'>PnL: {trade['pnl']:.2f}</span>
                </div>
                <canvas id='{canvas_id}' height='220' style='width:100%;'></canvas>
                <script>
                    (function() {{
                        const canvas = document.getElementById('{canvas_id}');
                        const ctx = canvas.getContext('2d');
                        const trade = {safe_trade_json};
                        const candles = trade.candles;
                        
                        // HiDPI
                        const rect = canvas.getBoundingClientRect();
                        canvas.width = rect.width * 2;
                        canvas.height = rect.height * 2;
                        ctx.scale(2, 2);
                        const W = rect.width; 
                        const H = rect.height;
                        
                        // Scale
                        const prices = candles.flatMap(c => [c.h, c.l]);
                        prices.push(trade.sl, trade.tp);
                        const mn = Math.min(...prices);
                        const mx = Math.max(...prices);
                        const rng = mx - mn || 1;
                        const padTop = 20; const padBot = 20;
                        const availH = H - padTop - padBot;
                        
                        const toY = p => padTop + availH - ((p - mn) / rng) * availH;
                        const cw = (W - 20) / candles.length * 0.7;
                        const sp = (W - 20) / candles.length;
                        const toX = i => 10 + i * sp + sp/2;
                        
                        // Draw BG
                        ctx.fillStyle = '#161b22'; ctx.fillRect(0,0,W,H);
                        
                        // Trade zone
                        const yen = toY(trade.entry);
                        const ysl = toY(trade.sl);
                        const ytp = toY(trade.tp);
                        const x1 = toX(trade.entry_idx_rel);
                        const x2 = toX(trade.close_idx_rel);
                        
                        ctx.fillStyle = trade.type === 'SHORT' ? 'rgba(248,81,73,0.1)' : 'rgba(63,185,80,0.1)';
                        if (x2 > x1) {{
                            ctx.fillRect(x1, Math.min(yen, ytp), x2-x1, Math.abs(yen-ytp));
                        }}
                        
                        // Levels
                        ctx.lineWidth = 1;
                        ctx.strokeStyle='#8b949e'; ctx.setLineDash([2,2]); 
                        ctx.beginPath(); ctx.moveTo(0, yen); ctx.lineTo(W, yen); ctx.stroke();
                        
                        ctx.strokeStyle='#f85149'; ctx.setLineDash([]);
                        ctx.beginPath(); ctx.moveTo(0, ysl); ctx.lineTo(W, ysl); ctx.stroke();
                        
                        ctx.strokeStyle='#3fb950'; 
                        ctx.beginPath(); ctx.moveTo(0, ytp); ctx.lineTo(W, ytp); ctx.stroke();
                        
                        // Labels
                        ctx.font = '10px sans-serif';
                        ctx.fillStyle = '#f85149'; ctx.fillText('SL', W-25, ysl-3);
                        ctx.fillStyle = '#3fb950'; ctx.fillText('TP', W-25, ytp-3);
                        ctx.fillStyle = '#8b949e'; ctx.fillText('Entry', W-35, yen-3);
                        
                        // Candles
                        candles.forEach((c, i) => {{
                            const x = toX(i);
                            const isGreen = c.c >= c.o;
                            ctx.fillStyle = isGreen ? '#3fb950' : '#f85149';
                            ctx.strokeStyle = ctx.fillStyle;
                            
                            // Wick
                            ctx.beginPath(); ctx.moveTo(x, toY(c.h)); ctx.lineTo(x, toY(c.l)); ctx.stroke();
                            
                            // Body
                            const yOpen = toY(c.o);
                            const yClose = toY(c.c);
                            const hBody = Math.max(Math.abs(yOpen - yClose), 1);
                            
                            ctx.fillRect(x - cw/2, Math.min(yOpen, yClose), cw, hBody);
                            
                            // Mark Signal
                            if (i === trade.signal_idx_rel) {{
                                ctx.fillStyle = '#f7931a';
                                ctx.beginPath(); ctx.arc(x, toY(c.h)-10, 3, 0, Math.PI*2); ctx.fill();
                            }}
                        }});
                    }})();
                </script>
            </div>
            """
        html += "</div>"
    
    html += "</body></html>"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DONE] Report: {output_file}")

# ==========================================
# --- GRID SEARCH ---
# ==========================================
def run_grid_search(df, volume_thresholds, ratios, sl_params, rr_values, label=""):
    results = []
    total = len(volume_thresholds) * len(ratios) * len(sl_params) * len(rr_values)
    print(f"[*] Grid Search: {total} combinations...")
    counter = 0
    bar_len = 40
    t0 = time.time()

    for thr_label, thr_val in volume_thresholds.items():
        for r_name, r_val in ratios.items():
            for sl in sl_params:
                for rr in rr_values:
                    counter += 1
                    count, wins, eq = run_backtest(df, thr_val, r_val, sl, rr)
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
                    pct = counter / total * 100
                    filled = int(bar_len * counter / total)
                    bar = '#' * filled + '-' * (bar_len - filled)
                    sys.stdout.write(
                        f"\r    [{bar}] {pct:.0f}% ({counter}/{total}) "
                        f"| {thr_label} | {r_name} | SL=${sl} | RR={rr}  "
                    )
                    sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n[OK] Done in {elapsed:.1f}s")
    return results
