"""
BTC Multi-TF Wick Anomaly Signal Bot
Monitors 15M, 30M, 1H, 4H for wick volume anomalies.
Sends Telegram signals with SL/TP, tracks trades, logs statistics.
"""

import time
import datetime
import json
import os
import sys
import requests
import pandas as pd
import numpy as np

from bot_config import (
    TELEGRAM_TOKEN, CHAT_ID, SYMBOLS, TRADE_LOG_FILE,
    INITIAL_CAPITAL, RISK_PER_TRADE, STRATEGIES
)

# ==========================================
# --- TELEGRAM HANDLER ---
# ==========================================
class TelegramBot:
    def __init__(self):
        self.base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        self.offset = 0

    def send_message(self, text):
        try:
            url = f"{self.base_url}/sendMessage"
            data = {"chat_id": CHAT_ID, "text": text}
            r = requests.post(url, data=data, timeout=5)
            if r.status_code != 200:
                print(f"[TG ERROR] Status: {r.status_code}, Response: {r.text}", flush=True)
            else:
                print(f"[TG] Message sent.", flush=True)
        except Exception as e:
            print(f"[TG ERROR] Send failed: {e}", flush=True)

    def get_updates(self):
        try:
            url = f"{self.base_url}/getUpdates"
            params = {"offset": self.offset, "timeout": 2}
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if "result" in data:
                    for u in data["result"]:
                        self.offset = u["update_id"] + 1
                        yield u
        except Exception as e:
            print(f"[TG ERROR] GetUpdates failed: {e}")

# ==========================================
# --- BINANCE DATA HANDLER ---
# ==========================================
class BinanceData:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"

    def get_recent_candles(self, symbol, limit=1500):
        """Fetch 1m candles from Binance Futures for a given symbol."""
        url = f"{self.base_url}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": "1m", "limit": limit}
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                candles = []
                for k in data:
                    ts = pd.to_datetime(k[0], unit='ms')
                    o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                    v = float(k[5])
                    tbv = float(k[9])
                    sell_vol = v - tbv
                    candles.append([ts, o, h, l, c, v, tbv, sell_vol])
                
                df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol','buy_vol','sell_vol'])
                return df
        except Exception as e:
            print(f"[DATA ERROR] {e}", flush=True)
        return pd.DataFrame()

    def get_historical_candles(self, symbol, days=30):
        """Fetch `days` worth of 1m candles by batching requests (max 1500 per call)."""
        all_candles = []
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
        batch_ms = 1500 * 60 * 1000  # 1500 minutes in ms
        
        cursor = start_ms
        url = f"{self.base_url}/fapi/v1/klines"
        batch_num = 0
        
        while cursor < now_ms:
            params = {
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "limit": 1500
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    if not data:
                        break
                    for k in data:
                        ts = pd.to_datetime(k[0], unit='ms')
                        o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                        v = float(k[5])
                        tbv = float(k[9])
                        sell_vol = v - tbv
                        all_candles.append([ts, o, h, l, c, v, tbv, sell_vol])
                    # Move cursor past last candle
                    cursor = int(data[-1][0]) + 60000
                else:
                    print(f"[DATA ERROR] HTTP {r.status_code} fetching history", flush=True)
                    break
            except Exception as e:
                print(f"[DATA ERROR] History fetch: {e}", flush=True)
                break
            
            batch_num += 1
            if batch_num % 10 == 0:
                print(f"  ... fetched {len(all_candles)} candles for {symbol}", flush=True)
            time.sleep(0.2)  # Rate limit
        
        if all_candles:
            df = pd.DataFrame(all_candles, columns=['ts','open','high','low','close','vol','buy_vol','sell_vol'])
            df = df.drop_duplicates(subset='ts').sort_values('ts').reset_index(drop=True)
            print(f"[DATA] Loaded {len(df)} 1m candles for {symbol} ({days} days)", flush=True)
            return df
        return pd.DataFrame()

    def get_ticker_price(self, symbol):
        try:
            url = f"{self.base_url}/fapi/v1/ticker/price"
            params = {"symbol": symbol}
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                return float(r.json()['price'])
        except Exception:
            pass
        return None

# ==========================================
# --- SIGNAL BOT CORE ---
# ==========================================
class SignalBot:
    def __init__(self):
        self.tg = TelegramBot()
        self.data = BinanceData()
        self.strategies = STRATEGIES
        self.equity = INITIAL_CAPITAL
        
        # State per strategy
        self.active_trades = {k: None for k in STRATEGIES}
        self.last_processed_ts = {k: None for k in STRATEGIES}
        
        # Trade history
        self.trade_history = []
        self.load_log()

    def load_log(self):
        if os.path.exists(TRADE_LOG_FILE):
            try:
                with open(TRADE_LOG_FILE, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.trade_history = data.get('trades', [])
                        self.equity = data.get('equity', INITIAL_CAPITAL)
                    else:
                        self.trade_history = data
                print(f"[*] Loaded {len(self.trade_history)} trades, equity: ${self.equity:.2f}")
            except:
                print("[!] Log file corrupted, starting fresh.")

    def save_log(self):
        data = {
            'equity': self.equity,
            'trades': self.trade_history
        }
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=4, default=str)

    def resample_and_calc_wick_volumes(self, df_1m, tf_str):
        """Resample 1m data to target TF and calculate wick volumes (identical to backtest logic)."""
        if df_1m.empty or len(df_1m) < 5:
            return pd.DataFrame()
        
        df_1m = df_1m.copy()
        
        # Resample OHLC
        df_tf = df_1m.set_index('ts').resample(tf_str).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna().reset_index()
        
        if len(df_tf) < 3:
            return pd.DataFrame()
        
        # Map 1m bars to their TF bar
        df_1m['tf_ts'] = df_1m['ts'].dt.floor(tf_str)
        
        # Merge TF open/close back to 1m
        merged = pd.merge(df_1m, df_tf[['ts','open','close']],
                          left_on='tf_ts', right_on='ts', suffixes=('','_tf'))
        
        merged['wick_top'] = merged[['open_tf','close_tf']].max(axis=1)
        merged['wick_bot'] = merged[['open_tf','close_tf']].min(axis=1)
        
        # Upper wick volumes
        mask_up = merged['close'] >= merged['wick_top']
        up_buy = merged[mask_up].groupby('tf_ts')['buy_vol'].sum()
        up_sell = merged[mask_up].groupby('tf_ts')['sell_vol'].sum()
        
        # Lower wick volumes
        mask_lo = merged['close'] <= merged['wick_bot']
        lo_buy = merged[mask_lo].groupby('tf_ts')['buy_vol'].sum()
        lo_sell = merged[mask_lo].groupby('tf_ts')['sell_vol'].sum()
        
        df_tf = df_tf.set_index('ts')
        df_tf['up_wick_buy'] = up_buy
        df_tf['up_wick_sell'] = up_sell
        df_tf['lo_wick_buy'] = lo_buy
        df_tf['lo_wick_sell'] = lo_sell
        df_tf.fillna(0, inplace=True)
        
        df_tf['up_wick_total'] = df_tf['up_wick_buy'] + df_tf['up_wick_sell']
        df_tf['lo_wick_total'] = df_tf['lo_wick_buy'] + df_tf['lo_wick_sell']
        
        return df_tf.reset_index()

    def calc_volume_threshold(self, df_tf, top_percent):
        """Calculate the volume threshold as percentile (same as backtest)."""
        up_nonzero = df_tf[df_tf['up_wick_total'] > 0]['up_wick_total']
        lo_nonzero = df_tf[df_tf['lo_wick_total'] > 0]['lo_wick_total']
        all_vols = pd.concat([up_nonzero, lo_nonzero])
        
        if len(all_vols) < 10:
            return float('inf')  # Not enough data
            
        return np.percentile(all_vols, 100 - top_percent)

    def check_signal(self, row, cfg, volume_threshold):
        """Check for anomaly signal on a completed candle."""
        imb = cfg['imbalance_ratio']
        
        is_short = (row['up_wick_total'] >= volume_threshold and
                    row['up_wick_sell'] > 0 and
                    row['up_wick_buy'] >= row['up_wick_sell'] * imb)
        
        is_long = (row['lo_wick_total'] >= volume_threshold and
                   row['lo_wick_buy'] > 0 and
                   row['lo_wick_sell'] >= row['lo_wick_buy'] * imb)
        
        # Double signal = skip
        if is_short and is_long:
            return None
        if is_short:
            return "SHORT"
        if is_long:
            return "LONG"
        return None

    def process_strategy(self, strat_key, df_1m):
        cfg = self.strategies[strat_key]
        tf_str = cfg['tf_str']
        symbol = cfg['symbol']
        
        # Resample and calculate wick volumes
        df_tf = self.resample_and_calc_wick_volumes(df_1m, tf_str)
        if df_tf.empty or len(df_tf) < 3:
            return
        
        # Calculate dynamic volume threshold from available history
        volume_threshold = self.calc_volume_threshold(df_tf, cfg['top_percent'])
        
        active_trade = self.active_trades[strat_key]
        last_candle = df_tf.iloc[-2]  # Last COMPLETED candle
        last_ts = last_candle['ts']
        
        current_price = self.data.get_ticker_price(symbol)
        if not current_price:
            current_price = df_tf.iloc[-1]['close']

        # --- SIGNAL CHECK (only if no active trade) ---
        if not active_trade:
            if self.last_processed_ts[strat_key] != str(last_ts):
                signal = self.check_signal(last_candle, cfg, volume_threshold)
                
                if signal:
                    entry_price = current_price
                    rr = cfg['rr']
                    buf = cfg['sl_buffer_pct']
                    
                    if signal == "SHORT":
                        sl_price = last_candle['high'] * (1 + buf)
                        dist = abs(sl_price - entry_price)
                        if dist < 1:
                            sl_price = entry_price + (entry_price * 0.001)
                            dist = abs(sl_price - entry_price)
                        tp_price = entry_price - (dist * rr)
                    else:  # LONG
                        sl_price = last_candle['low'] * (1 - buf)
                        dist = abs(entry_price - sl_price)
                        if dist < 1:
                            sl_price = entry_price - (entry_price * 0.001)
                            dist = abs(entry_price - sl_price)
                        tp_price = entry_price + (dist * rr)
                    
                    # Position sizing: 1% risk
                    risk_amt = self.equity * RISK_PER_TRADE
                    size_btc = risk_amt / dist
                    
                    new_trade = {
                        "id": f"{strat_key}_{int(time.time())}",
                        "strategy": strat_key,
                        "direction": signal,
                        "entry": round(entry_price, 2),
                        "sl": round(sl_price, 2),
                        "tp": round(tp_price, 2),
                        "size_btc": round(size_btc, 6),
                        "risk_usd": round(risk_amt, 2),
                        "rr": rr,
                        "open_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "candle_ts": str(last_ts),
                        "volume_threshold": round(volume_threshold, 2),
                        "up_wick_buy": round(last_candle['up_wick_buy'], 2),
                        "up_wick_sell": round(last_candle['up_wick_sell'], 2),
                        "lo_wick_buy": round(last_candle['lo_wick_buy'], 2),
                        "lo_wick_sell": round(last_candle['lo_wick_sell'], 2),
                    }
                    
                    self.active_trades[strat_key] = new_trade
                    self.last_processed_ts[strat_key] = str(last_ts)
                    
                    # Telegram notification
                    sl_dist_pct = abs(sl_price - entry_price) / entry_price * 100
                    tp_dist_pct = abs(tp_price - entry_price) / entry_price * 100
                    emoji = "🔴" if signal == "SHORT" else "🟢"
                    
                    msg = (f"⚡️ {emoji} SIGNAL: {signal} ⚡️\n"
                           f"Strategy: {cfg['name']}\n"
                           f"━━━━━━━━━━━━━━━\n"
                           f"Entry: ${entry_price:,.2f}\n"
                           f"SL: ${sl_price:,.2f} ({sl_dist_pct:.2f}%)\n"
                           f"TP: ${tp_price:,.2f} ({tp_dist_pct:.2f}%)\n"
                           f"RR: 1:{rr}")
                    self.tg.send_message(msg)
                    print(f"\n[SIGNAL] {strat_key}: {signal} @ ${entry_price:,.2f} | SL ${sl_price:,.2f} | TP ${tp_price:,.2f}", flush=True)
                else:
                    self.last_processed_ts[strat_key] = str(last_ts)

        # --- TRADE MANAGEMENT ---
        else:
            trade = active_trade
            direction = trade['direction']
            sl = trade['sl']
            tp = trade['tp']
            entry = trade['entry']
            size = trade['size_btc']
            
            close_reason = None
            exit_price = current_price
            
            if direction == "LONG":
                if current_price >= tp:
                    close_reason = "TP ✅"
                    exit_price = tp
                elif current_price <= sl:
                    close_reason = "SL ❌"
                    exit_price = sl
            else:  # SHORT
                if current_price <= tp:
                    close_reason = "TP ✅"
                    exit_price = tp
                elif current_price >= sl:
                    close_reason = "SL ❌"
                    exit_price = sl
            
            if close_reason:
                # Calculate PnL
                if direction == "LONG":
                    gross_pnl = (exit_price - entry) * size
                else:
                    gross_pnl = (entry - exit_price) * size
                
                # Commission + slippage
                comm = (entry * size * 0.0004) + (exit_price * size * 0.0004)
                slip = (entry * size * 0.0001) + (exit_price * size * 0.0001)
                net_pnl = gross_pnl - comm - slip
                
                self.equity += net_pnl
                
                result = "WIN" if net_pnl > 0 else "LOSS"
                
                # Record trade
                trade['close_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade['close_price'] = round(exit_price, 2)
                trade['close_reason'] = close_reason.replace(" ✅", "").replace(" ❌", "")
                trade['gross_pnl'] = round(gross_pnl, 2)
                trade['commission'] = round(comm, 2)
                trade['slippage'] = round(slip, 2)
                trade['net_pnl'] = round(net_pnl, 2)
                trade['result'] = result
                trade['equity_after'] = round(self.equity, 2)
                
                self.trade_history.append(trade)
                self.save_log()
                self.active_trades[strat_key] = None
                
                # Statistics
                total = len(self.trade_history)
                wins = sum(1 for t in self.trade_history if t['result'] == 'WIN')
                total_pnl = sum(t['net_pnl'] for t in self.trade_history)
                wr = (wins / total * 100) if total > 0 else 0
                
                emoji = "✅" if result == "WIN" else "❌"
                msg = (f"{emoji} TRADE CLOSED ({cfg['name']})\n"
                       f"━━━━━━━━━━━━━━━\n"
                       f"Direction: {direction}\n"
                       f"Entry: ${entry:,.2f}\n"
                       f"Exit: ${exit_price:,.2f} ({close_reason})\n"
                       f"Net PnL: ${net_pnl:,.2f}")
                self.tg.send_message(msg)
                print(f"\n[CLOSED] {strat_key}: {direction} {close_reason} | PnL: ${net_pnl:.2f} | Equity: ${self.equity:.2f}", flush=True)

    def run_monthly_scan(self):
        """Scan last 30 days across all strategies, simulate every signal."""
        print("\n[HISTORY] Starting 30-day scan...", flush=True)
        
        # 1. Fetch 30 days of 1m data per symbol
        symbol_data = {}
        for sym in SYMBOLS:
            df = self.data.get_historical_candles(sym, days=30)
            if not df.empty:
                symbol_data[sym] = df
        
        if not symbol_data:
            return []
        
        all_trades = []
        
        # 2. Process each strategy
        for strat_key, cfg in self.strategies.items():
            sym = cfg['symbol']
            if sym not in symbol_data:
                continue
            
            df_1m = symbol_data[sym]
            tf_str = cfg['tf_str']
            
            # Resample to strategy TF
            df_tf = self.resample_and_calc_wick_volumes(df_1m, tf_str)
            if df_tf.empty or len(df_tf) < 10:
                continue
            
            volume_threshold = self.calc_volume_threshold(df_tf, cfg['top_percent'])
            
            # Walk through each completed candle
            in_trade = False
            trade_entry = None
            trade_sl = None
            trade_tp = None
            trade_dir = None
            trade_ts = None
            
            for i in range(1, len(df_tf) - 1):
                row = df_tf.iloc[i]
                
                # If in a trade, check SL/TP on subsequent TF candles
                if in_trade:
                    h = row['high']
                    l = row['low']
                    
                    if trade_dir == "LONG":
                        if l <= trade_sl:
                            all_trades.append({
                                'strategy': strat_key,
                                'name': cfg['name'],
                                'direction': trade_dir,
                                'entry': trade_entry,
                                'sl': trade_sl,
                                'tp': trade_tp,
                                'exit': trade_sl,
                                'result': 'SL',
                                'pnl_pct': -abs(trade_sl - trade_entry) / trade_entry * 100,
                                'ts': str(trade_ts),
                                'exit_ts': str(row['ts']),
                            })
                            in_trade = False
                            continue
                        elif h >= trade_tp:
                            all_trades.append({
                                'strategy': strat_key,
                                'name': cfg['name'],
                                'direction': trade_dir,
                                'entry': trade_entry,
                                'sl': trade_sl,
                                'tp': trade_tp,
                                'exit': trade_tp,
                                'result': 'TP',
                                'pnl_pct': abs(trade_tp - trade_entry) / trade_entry * 100,
                                'ts': str(trade_ts),
                                'exit_ts': str(row['ts']),
                            })
                            in_trade = False
                            continue
                    else:  # SHORT
                        if h >= trade_sl:
                            all_trades.append({
                                'strategy': strat_key,
                                'name': cfg['name'],
                                'direction': trade_dir,
                                'entry': trade_entry,
                                'sl': trade_sl,
                                'tp': trade_tp,
                                'exit': trade_sl,
                                'result': 'SL',
                                'pnl_pct': -abs(trade_sl - trade_entry) / trade_entry * 100,
                                'ts': str(trade_ts),
                                'exit_ts': str(row['ts']),
                            })
                            in_trade = False
                            continue
                        elif l <= trade_tp:
                            all_trades.append({
                                'strategy': strat_key,
                                'name': cfg['name'],
                                'direction': trade_dir,
                                'entry': trade_entry,
                                'sl': trade_sl,
                                'tp': trade_tp,
                                'exit': trade_tp,
                                'result': 'TP',
                                'pnl_pct': abs(trade_entry - trade_tp) / trade_entry * 100,
                                'ts': str(trade_ts),
                                'exit_ts': str(row['ts']),
                            })
                            in_trade = False
                            continue
                    continue  # Still in trade, skip signal check
                
                # Check for new signal on completed candle
                signal = self.check_signal(row, cfg, volume_threshold)
                if signal:
                    # Entry = next candle open
                    next_row = df_tf.iloc[i + 1]
                    entry_price = next_row['open']
                    buf = cfg['sl_buffer_pct']
                    rr = cfg['rr']
                    
                    if signal == "SHORT":
                        sl_price = row['high'] * (1 + buf)
                        dist = abs(sl_price - entry_price)
                        if dist < entry_price * 0.0001:
                            continue
                        tp_price = entry_price - (dist * rr)
                    else:  # LONG
                        sl_price = row['low'] * (1 - buf)
                        dist = abs(entry_price - sl_price)
                        if dist < entry_price * 0.0001:
                            continue
                        tp_price = entry_price + (dist * rr)
                    
                    in_trade = True
                    trade_entry = round(entry_price, 2)
                    trade_sl = round(sl_price, 2)
                    trade_tp = round(tp_price, 2)
                    trade_dir = signal
                    trade_ts = row['ts']
            
            print(f"[HISTORY] {strat_key}: found {sum(1 for t in all_trades if t['strategy'] == strat_key)} trades", flush=True)
        
        # 3. Mark which trades were taken by the bot
        bot_candle_ts = set()
        for t in self.trade_history:
            if 'candle_ts' in t:
                bot_candle_ts.add((t['strategy'], t['candle_ts']))
        
        for t in all_trades:
            key = (t['strategy'], t['ts'])
            t['taken_by_bot'] = key in bot_candle_ts
        
        # 4. Save to file
        history_file = os.path.join(os.path.dirname(TRADE_LOG_FILE) or '.', 'monthly_history.json')
        with open(history_file, 'w') as f:
            json.dump(all_trades, f, indent=2, default=str)
        
        print(f"[HISTORY] Scan complete: {len(all_trades)} total trades found", flush=True)
        return all_trades

    def handle_commands(self):
        for update in self.tg.get_updates():
            if "message" not in update:
                continue
            msg = update["message"]
            text = msg.get("text", "")
            
            if text == "/stats":
                total = len(self.trade_history)
                wins = sum(1 for t in self.trade_history if t['result'] == 'WIN')
                losses = total - wins
                wr = (wins / total * 100) if total else 0
                total_pnl = sum(t['net_pnl'] for t in self.trade_history)
                
                # Per-strategy breakdown
                strat_stats = ""
                for sk in self.strategies:
                    st = [t for t in self.trade_history if t['strategy'] == sk]
                    if st:
                        sw = sum(1 for t in st if t['result'] == 'WIN')
                        sp = sum(t['net_pnl'] for t in st)
                        swr = (sw / len(st) * 100)
                        strat_stats += f"\n  {sk}: {len(st)} trades, WR {swr:.0f}%, PnL ${sp:.2f}"
                    else:
                        strat_stats += f"\n  {sk}: No trades yet"
                
                reply = (f"📊 BOT STATISTICS\n"
                         f"━━━━━━━━━━━━━━━\n"
                         f"Equity: ${self.equity:,.2f}\n"
                         f"Total Trades: {total}\n"
                         f"Wins: {wins} | Losses: {losses}\n"
                         f"Win Rate: {wr:.1f}%\n"
                         f"Total PnL: ${total_pnl:,.2f}\n"
                         f"━━━━━━━━━━━━━━━\n"
                         f"Per Strategy:{strat_stats}")
                self.tg.send_message(reply)
                
            elif text == "/status":
                active = [(k, t) for k, t in self.active_trades.items() if t]
                if not active:
                    self.tg.send_message("💤 No active trades.")
                else:
                    reply = "🔥 ACTIVE TRADES:\n"
                    for sk, t in active:
                        sym = self.strategies[sk]['symbol']
                        price = self.data.get_ticker_price(sym) or t['entry']
                        if t['direction'] == "LONG":
                            cur_pnl = (price - t['entry']) * t['size_btc']
                        else:
                            cur_pnl = (t['entry'] - price) * t['size_btc']
                        emoji = "🟢" if cur_pnl >= 0 else "🔴"
                        reply += (f"\n{emoji} {sk} ({t['direction']})\n"
                                  f"  Entry: ${t['entry']:,.2f}\n"
                                  f"  SL: ${t['sl']:,.2f} | TP: ${t['tp']:,.2f}\n"
                                  f"  PnL: ${cur_pnl:,.2f}\n")
                    self.tg.send_message(reply)
                    
            elif text == "/equity":
                self.tg.send_message(f"💰 Current Equity: ${self.equity:,.2f}")
            
            elif text == "/history":
                self.tg.send_message("⏳ Scanning 30 days of data... Please wait ~30 sec.")
                try:
                    trades = self.run_monthly_scan()
                    
                    if not trades:
                        self.tg.send_message("❌ No data available for scan.")
                        continue
                    
                    total = len(trades)
                    wins = sum(1 for t in trades if t['result'] == 'TP')
                    losses = sum(1 for t in trades if t['result'] == 'SL')
                    wr = (wins / total * 100) if total else 0
                    taken = sum(1 for t in trades if t.get('taken_by_bot'))
                    missed = total - taken
                    
                    # Simulated PnL (1% risk per trade on $10k)
                    sim_pnl = 0
                    for t in trades:
                        if t['result'] == 'TP':
                            # Find strategy RR
                            rr = self.strategies.get(t['strategy'], {}).get('rr', 2)
                            sim_pnl += INITIAL_CAPITAL * RISK_PER_TRADE * rr
                        else:
                            sim_pnl -= INITIAL_CAPITAL * RISK_PER_TRADE
                    
                    # Per-strategy breakdown
                    strat_lines = ""
                    for sk in self.strategies:
                        st = [t for t in trades if t['strategy'] == sk]
                        if st:
                            sw = sum(1 for t in st if t['result'] == 'TP')
                            swr = (sw / len(st) * 100)
                            strat_lines += f"\n  {self.strategies[sk]['name']}: {len(st)} trades, WR {swr:.0f}%"
                        else:
                            strat_lines += f"\n  {self.strategies[sk]['name']}: 0 trades"
                    
                    summary = (f"📅 HISTORY (30 days)\n"
                               f"━━━━━━━━━━━━━━━\n"
                               f"Total signals: {total}\n"
                               f"Taken by bot: {taken}\n"
                               f"Missed: {missed}\n"
                               f"━━━━━━━━━━━━━━━\n"
                               f"Wins: {wins} | Losses: {losses}\n"
                               f"Win Rate: {wr:.1f}%\n"
                               f"Sim. PnL: ${sim_pnl:+,.2f}\n"
                               f"━━━━━━━━━━━━━━━\n"
                               f"Per Strategy:{strat_lines}")
                    self.tg.send_message(summary)
                    
                    # Last 10 trades
                    recent = trades[-10:]
                    if recent:
                        detail = "Last 10 signals:\n"
                        for j, t in enumerate(reversed(recent), 1):
                            emoji = "🟢" if t['direction'] == "LONG" else "🔴"
                            res_emoji = "✅" if t['result'] == 'TP' else "❌"
                            ts_short = t['ts'][:16]  # YYYY-MM-DD HH:MM
                            bot_tag = "" if t.get('taken_by_bot') else " [MISSED]"
                            detail += (f"{j}. {emoji} {t['direction']} {t['name']} | {ts_short}\n"
                                       f"   Entry ${t['entry']:,.2f} → {t['result']} {res_emoji} ({t['pnl_pct']:+.2f}%){bot_tag}\n")
                        self.tg.send_message(detail)
                
                except Exception as e:
                    print(f"[ERROR] History scan failed: {e}", flush=True)
                    self.tg.send_message(f"❌ Scan error: {e}")

    def run(self):
        print(f"[*] Bot Started. Strategies: {list(self.strategies.keys())}", flush=True)
        
        # Startup message
        strats_info = ""
        for sk, cfg in self.strategies.items():
            strats_info += f"\n  • {cfg['name']} (Top {cfg['top_percent']}%, Imb {cfg['imbalance_ratio']}x, RR 1:{cfg['rr']})"
        
        symbols_str = ', '.join(SYMBOLS)
        self.tg.send_message(
            f"🚀 BOT STARTED\n"
            f"Symbols: {symbols_str}\n"
            f"Equity: ${self.equity:,.2f}\n"
            f"Risk/Trade: {RISK_PER_TRADE*100:.0f}%\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Strategies ({len(self.strategies)}):{strats_info}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Commands: /stats /status /equity /history"
        )
        
        while True:
            try:
                # 1. Handle Telegram commands
                self.handle_commands()
                
                # 2. Fetch 1m data per symbol (cache to avoid redundant requests)
                symbol_data = {}
                for sym in SYMBOLS:
                    df = self.data.get_recent_candles(sym, limit=1500)
                    if not df.empty:
                        symbol_data[sym] = df
                
                # 3. Process each strategy with its symbol's data
                for s_name, cfg in self.strategies.items():
                    sym = cfg['symbol']
                    if sym in symbol_data:
                        self.process_strategy(s_name, symbol_data[sym])
                
                # Heartbeat
                now = datetime.datetime.now().strftime("%H:%M:%S")
                active_count = sum(1 for t in self.active_trades.values() if t)
                sys.stdout.write(f"\r[*] {now} | Equity: ${self.equity:,.2f} | Active: {active_count} | Strategies: {len(self.strategies)}   ")
                sys.stdout.flush()

                # Sleep 15 seconds between cycles
                time.sleep(15)
                
            except KeyboardInterrupt:
                print("\n[!] Stopping bot...", flush=True)
                self.tg.send_message("⛔ BOT STOPPED")
                break
            except Exception as e:
                print(f"\n[ERROR] Loop crash: {e}", flush=True)
                time.sleep(15)

if __name__ == "__main__":
    try:
        print("=" * 50, flush=True)
        print("  BTC Multi-TF Wick Anomaly Signal Bot", flush=True)
        print("=" * 50, flush=True)
        bot = SignalBot()
        bot.run()
    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        input("Press Enter to exit...")
