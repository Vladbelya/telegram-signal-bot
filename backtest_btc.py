import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from datetime import time

def load_or_generate_data(filepath="data.csv"):
    """
    Загружаем данные из CSV. Если файла нет, генерируем синтетические (случайные блуждания) 
    1-минутные данные за 60 дней для проверки работоспособности скрипта.
    """
    if os.path.exists(filepath):
        print(f"[INFO] Загрузка исторических данных из файла {filepath}...")
        df = pd.read_csv(filepath)
        
        time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        
        # Конвертация в US/Eastern с учетом переходов времени 
        # (предполагаем, что исходные данные в UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')
        return df
    else:
        print(f"[WARN] Файл {filepath} не найден. Генерируем 60 дней синтетических данных BTC для теста...")
        now_utc = pd.Timestamp.utcnow()
        start_time = now_utc - pd.Timedelta(days=60)
        idx = pd.date_range(start=start_time, end=now_utc, freq='1min')
        
        np.random.seed(42)  # для воспроизводимости
        returns = np.random.normal(0, 0.0005, size=len(idx))
        close_prices = 50000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.0001, size=len(idx))),
            'High': close_prices * (1 + abs(np.random.normal(0, 0.0002, size=len(idx)))),
            'Low': close_prices * (1 - abs(np.random.normal(0, 0.0002, size=len(idx)))),
            'Close': close_prices
        }, index=idx)
        
        df.index = df.index.tz_convert('US/Eastern')
        return df

def backtest_strategy(df: pd.DataFrame, evaluation_period: int, risk_reward_ratio: float):
    # Оставляем только внутридневную сессию 09:30 - 16:00 ET
    df_session = df.between_time('09:30', '16:00')
    days = df_session.groupby(df_session.index.date)
    
    capital = 10000.0
    capital_history = []
    trades = []
    
    for date, day_data in days:
        # Убедимся, что данных хватает хотя бы для свечи оценки
        if len(day_data) < evaluation_period + 1:
            capital_history.append({'Date': date, 'Capital': capital})
            continue
            
        start_idx = 0
        trade_taken = False
        
        while start_idx + evaluation_period < len(day_data) and not trade_taken:
            # ресемплинг evaluate_period свечи из минуток
            eval_bars = day_data.iloc[start_idx : start_idx + evaluation_period]
            
            signal_open = eval_bars['Open'].iloc[0]
            signal_close = eval_bars['Close'].iloc[-1]
            signal_high = eval_bars['High'].max()
            signal_low = eval_bars['Low'].min()
            
            # Индекс входа: следующая 1м свеча
            entry_idx = start_idx + evaluation_period
            
            # Выход за пределы сессии?
            if entry_idx >= len(day_data):
                break
                
            entry_price = day_data['Open'].iloc[entry_idx]
            
            # Определение направления
            if signal_close > signal_open:
                is_long = True
                sl_price = signal_low
            elif signal_close < signal_open:
                is_long = False
                sl_price = signal_high
            else:
                # Доджи - игнорируем, сдвигаемся на evaluation_period
                start_idx += evaluation_period
                continue
                
            dist_to_sl = abs(entry_price - sl_price)
            if dist_to_sl == 0:
                # Ситуация на синтетике/плохих данных
                start_idx += evaluation_period
                continue
                
            # Риск-менеджмент: строго 1% от текущего капитала
            risk_amount = capital * 0.01
            size = risk_amount / dist_to_sl  # размер в базовом активе (BTC)
            
            if is_long:
                tp_price = entry_price + (dist_to_sl * risk_reward_ratio)
            else:
                tp_price = entry_price - (dist_to_sl * risk_reward_ratio)
                
            exit_price = None
            exit_slip = 0.0
            exit_fee = 0.0
            exit_type = ""
            
            # Проверяем исполнение каждой 1-минутной свечой до конца сессии
            for j in range(entry_idx, len(day_data)):
                bar = day_data.iloc[j]
                
                low = bar['Low']
                high = bar['High']
                
                is_sl_hit = False
                is_tp_hit = False
                
                if is_long:
                    if low <= sl_price: is_sl_hit = True
                    if high >= tp_price: is_tp_hit = True
                else:
                    if high >= sl_price: is_sl_hit = True
                    if low <= tp_price: is_tp_hit = True

                # Если цена коснулась и SL, и TP за одну 1м свечу - консервативно берем SL
                if is_sl_hit and is_tp_hit:
                    exit_price = sl_price
                    exit_slip = 0.0001
                    exit_fee = 0.0004
                    exit_type = "SL"
                    break
                elif is_sl_hit:
                    exit_price = sl_price
                    exit_slip = 0.0001
                    exit_fee = 0.0004
                    exit_type = "SL"
                    break
                elif is_tp_hit:
                    exit_price = tp_price
                    exit_slip = 0.0
                    exit_fee = 0.0002  # Лимиткой, мейкер фи 0.02%
                    exit_type = "TP"
                    break
                
                # Если 16:00, закрываем рыночным по Close текущего 1М бара
                if bar.name.time() >= time(16, 0):
                    exit_price = bar['Close']
                    exit_slip = 0.0001
                    exit_fee = 0.0004
                    exit_type = "Time (16:00)"
                    break
                    
            if exit_price is None:
                # На случай, если в данных нет свечи 16:00, закрываем на последней доступной
                exit_price = day_data.iloc[-1]['Close']
                exit_slip = 0.0001
                exit_fee = 0.0004
                exit_type = "EOD"
                
            # Расчет прибыли и комиссий
            entry_slip_rate = 0.0001
            entry_fee_rate = 0.0004
            
            # Исполнение с проскальзыванием
            if is_long:
                entry_exec = entry_price * (1 + entry_slip_rate)
                exit_exec = exit_price * (1 - exit_slip)
                gross_pnl = (exit_exec - entry_exec) * size
            else:
                entry_exec = entry_price * (1 - entry_slip_rate)
                exit_exec = exit_price * (1 + exit_slip)
                gross_pnl = (entry_exec - exit_exec) * size
                
            entry_comm = size * entry_price * entry_fee_rate
            exit_comm = size * exit_price * exit_fee
            
            net_pnl = gross_pnl - entry_comm - exit_comm
            capital += net_pnl
            
            trades.append({
                'Date': date,
                'Direction': 'Long' if is_long else 'Short',
                'Entry Time': day_data.index[entry_idx],
                'Entry Price': round(entry_price, 2),
                'SL': round(sl_price, 2),
                'TP': round(tp_price, 2),
                'Size': size,
                'Exit Price': round(exit_price, 2),
                'Exit Type': exit_type,
                'Net PnL': net_pnl,
                'Capital': capital
            })
            
            trade_taken = True
            
        capital_history.append({'Date': date, 'Capital': capital})
            
    return trades, capital_history

def run_grid_search(df):
    evaluation_periods = [5, 10, 15, 30, 60]
    risk_reward_ratios = [1.0, 2.0, 3.0]
    
    results = []
    best_combination = None
    best_net_profit = -float('inf')
    best_capital_history = None
    
    print("[INFO] Выполнение Grid Search параметров...")
    
    for eval_period, rr_ratio in product(evaluation_periods, risk_reward_ratios):
        trades, cap_hist = backtest_strategy(df, eval_period, rr_ratio)
        
        total_trades = len(trades)
        if total_trades == 0:
            results.append({
                'Evaluation Period': eval_period, 'R:R': rr_ratio, 
                'Total Trades': 0, 'Winrate %': 0.0, 'Net Profit ($)': 0.0,
                'Max Drawdown (%)': 0.0, 'Profit Factor': 0.0
            })
            continue
            
        winning_trades = sum(1 for t in trades if t['Net PnL'] > 0)
        winrate = (winning_trades / total_trades) * 100
        
        final_capital = cap_hist[-1]['Capital']
        net_profit = final_capital - 10000.0
        
        # Max Drawdown
        curve = pd.Series([10000.0] + [c['Capital'] for c in cap_hist])
        running_max = curve.cummax()
        drawdowns = (curve - running_max) / running_max
        max_dd = drawdowns.min() * 100 
        
        # Profit Factor
        gross_profits = sum(t['Net PnL'] for t in trades if t['Net PnL'] > 0)
        gross_losses = sum(t['Net PnL'] for t in trades if t['Net PnL'] < 0)
        profit_factor = gross_profits / abs(gross_losses) if gross_losses != 0 else np.inf
        
        results.append({
            'Evaluation Period': eval_period,
            'R:R': rr_ratio,
            'Total Trades': total_trades,
            'Winrate %': round(winrate, 2),
            'Net Profit ($)': round(net_profit, 2),
            'Max Drawdown (%)': round(max_dd, 2),
            'Profit Factor': round(profit_factor, 2)
        })
        
        if net_profit > best_net_profit:
            best_net_profit = net_profit
            best_combination = (eval_period, rr_ratio)
            best_capital_history = cap_hist
            
    results_df = pd.DataFrame(results)
    
    print("\n--- ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ---")
    print(results_df.to_string(index=False))
    
    if best_capital_history and best_combination:
        dates = [c['Date'] for c in best_capital_history]
        caps = [c['Capital'] for c in best_capital_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, caps, marker='.', linestyle='-', color='b', linewidth=1)
        
        eval_p, rr = best_combination
        plt.title(f"Equity Curve - Best Params: Eval {eval_p}m, R:R {rr}")
        plt.xlabel("Date")
        plt.ylabel("Capital ($)")
        plt.grid(True, alpha=0.3)
        
        plot_path = "best_equity_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[INFO] Кривая доходности лучшей комбинации (Eval={eval_p}, R:R={rr}) сохранена в '{os.path.abspath(plot_path)}'")
        
if __name__ == "__main__":
    # Разместите Ваш файл с данными рядом со скриптом и назовите data.csv
    # Либо он сгенерирует синтетику для демонстрации
    df = load_or_generate_data("data.csv")
    run_grid_search(df)
