# 30M Wick Volume Anomaly Strategy — Full Specification

## Overview
A mean-reversion strategy that identifies abnormal volume concentrations in candle wicks on the 30-minute timeframe of BTCUSDT Perpetual Futures. The strategy detects exhaustion of aggressive participants (buyers in upper wicks, sellers in lower wicks) and trades the expected reversal.

## Instrument & Timeframe
- **Symbol**: BTCUSDT Binance Perpetual Futures
- **Analysis Timeframe**: 30-minute candles
- **Execution Timeframe**: Monitored on 1-minute data for precise SL/TP tracking
- **Trading Hours**: 24/7 (no session filter)

---

## Step 1: Data Preparation (Footprint Construction)

### 1.1 Collect 1-minute data
Fetch the last 1500 one-minute OHLCV candles from Binance Futures, including `taker_buy_volume`.

### 1.2 Calculate Sell Volume
```
sell_vol = total_vol - taker_buy_vol
```

### 1.3 Resample to 30-minute OHLC
Aggregate 1-minute candles into 30-minute candles:
- Open = first 1m open
- High = max of all 1m highs
- Low = min of all 1m lows  
- Close = last 1m close

### 1.4 Determine Candle Body Boundaries
For each 30m candle:
```
body_top = max(open_30m, close_30m)
body_bot = min(open_30m, close_30m)
```

### 1.5 Classify 1-minute Bars into Wick Zones
For each 1-minute bar within a 30-minute candle:
- If `1m_close >= body_top` → this bar belongs to the **Upper Wick**
- If `1m_close <= body_bot` → this bar belongs to the **Lower Wick**
- Otherwise → this bar belongs to the **Body** (ignored)

### 1.6 Aggregate Wick Volumes
For each 30m candle, sum up:
```
up_wick_buy_vol  = SUM(taker_buy_vol)  where 1m bar is in Upper Wick
up_wick_sell_vol = SUM(sell_vol)        where 1m bar is in Upper Wick
lo_wick_buy_vol  = SUM(taker_buy_vol)  where 1m bar is in Lower Wick
lo_wick_sell_vol = SUM(sell_vol)        where 1m bar is in Lower Wick

up_wick_total = up_wick_buy_vol + up_wick_sell_vol
lo_wick_total = lo_wick_buy_vol + lo_wick_sell_vol
```

---

## Step 2: Anomaly Detection (Dynamic Percentile Threshold)

### 2.1 Calculate Volume Threshold
From all available 30m candles in the history:
1. Collect all non-zero `up_wick_total` values
2. Collect all non-zero `lo_wick_total` values
3. Combine into one array
4. Calculate the **96th percentile** (Top 4%)

```
volume_threshold = percentile(all_nonzero_wick_volumes, 96)
```

This threshold adapts dynamically to market conditions.

---

## Step 3: Signal Generation

### 3.1 SHORT Signal (Upper Wick Anomaly)
All three conditions must be TRUE on the **last completed** 30m candle:

```
1. up_wick_total >= volume_threshold     (anomalous volume in upper wick)
2. up_wick_buy_vol >= 2.0 * up_wick_sell_vol   (buy imbalance ≥ 2x)
3. up_wick_sell_vol > 0                  (prevent division by zero)
```

**Interpretation**: Aggressive buyers pushed price into the upper wick with 2x more volume than sellers, but price was rejected back into the body. This signals buyer exhaustion → expect price to drop.

### 3.2 LONG Signal (Lower Wick Anomaly)
All three conditions must be TRUE on the **last completed** 30m candle:

```
1. lo_wick_total >= volume_threshold     (anomalous volume in lower wick)
2. lo_wick_sell_vol >= 2.0 * lo_wick_buy_vol   (sell imbalance ≥ 2x)
3. lo_wick_buy_vol > 0                   (prevent division by zero)
```

**Interpretation**: Aggressive sellers pushed price into the lower wick with 2x more volume than buyers, but price was rejected back into the body. This signals seller exhaustion → expect price to rise.

### 3.3 Double Signal Filter
If BOTH short and long conditions trigger on the same candle → **SKIP** (ambiguous signal, no trade).

---

## Step 4: Entry Execution

- **Timing**: Enter at market price immediately after the signal candle closes
- **Entry Price**: Current market price at the moment of detection
- No partial fills, no limit orders — market execution assumed

---

## Step 5: Stop Loss Placement

### 5.1 SHORT Stop Loss
```
SL = high_of_signal_candle × (1 + 0.0005)
```
SL is placed **above the high** of the anomalous 30m candle + 0.05% buffer.

### 5.2 LONG Stop Loss  
```
SL = low_of_signal_candle × (1 - 0.0005)
```
SL is placed **below the low** of the anomalous 30m candle + 0.05% buffer.

### 5.3 Rationale
The wick extreme represents the point where aggressive participants were absorbed. If price breaks beyond this level + buffer, the anomaly thesis is invalidated.

---

## Step 6: Take Profit Placement

### 6.1 Calculate Distance to SL
```
dist_to_sl = abs(entry_price - sl_price)
```

### 6.2 Set TP at R:R 2.0
```
SHORT: tp_price = entry_price - (dist_to_sl × 2.0)
LONG:  tp_price = entry_price + (dist_to_sl × 2.0)
```

Fixed R:R ratio of 1:2.0 — for every $1 risked, target $2 profit.

---

## Step 7: Position Sizing

### 7.1 Risk Per Trade
```
risk_amount = current_equity × 0.01   (1% of capital)
```

### 7.2 Position Size
```
size_btc = risk_amount / dist_to_sl
```

Example: If equity = $10,000, dist_to_sl = $530:
```
risk = $10,000 × 0.01 = $100
size = $100 / $530 = 0.1887 BTC
```

---

## Step 8: Trade Management

### 8.1 Hold Until Outcome
- Trade remains open until either **SL** or **TP** is hit
- No trailing stop, no time-based exit, no break-even moves
- No manual intervention

### 8.2 Conflict Resolution
If both SL and TP are hit on the same 1-minute candle → assume **SL hit first** (conservative).

### 8.3 Sequential Trades
After a trade closes, the next signal can only come from a 30m candle that **starts after** the exit candle. No re-entry on the same candle.

---

## Step 9: Cost Model

### 9.1 Commission
```
commission = (entry_price × size × 0.04%) + (exit_price × size × 0.04%)
```
Assumes taker fees on both entry and exit (0.04% each way = 0.08% round trip).

### 9.2 Slippage
```
slippage = (entry_price × size × 0.01%) + (exit_price × size × 0.01%)
```

### 9.3 Net PnL
```
net_pnl = gross_pnl - commission - slippage
```

---

## Backtest Results (BTCUSDT, Jan 2025 — Dec 2025)

| Metric | Value |
|--------|-------|
| **Initial Capital** | $10,000 |
| **Net Profit** | +$569.25 (+5.69%) |
| **Total Trades** | 10 |
| **Win Rate** | 60.0% (6W / 4L) |
| **Profit Factor** | 2.22 |
| **Max Drawdown** | 2.91% |
| **Risk Per Trade** | 1% |
| **R:R Ratio** | 1:2.0 |

---

## Optimal Parameters Summary

| Parameter | Value |
|-----------|-------|
| Timeframe | 30M |
| Volume Threshold | Top 4% (96th percentile) |
| Imbalance Ratio | 2.0x |
| Risk:Reward | 1:2.0 |
| SL Buffer | 0.05% behind High/Low |
| Risk Per Trade | 1% of equity |
| Commission | 0.04% taker each way |
| Slippage | 0.01% each way |

---

## Key Edge Explanation

The strategy exploits a specific market microstructure phenomenon:

1. **Aggressive participants enter at extremes** — high buy volume in upper wicks means aggressive market buyers are absorbing asks at premium prices
2. **Price rejection = absorption complete** — the wick forming back into the body means limit sellers absorbed all aggressive buying
3. **Post-absorption direction** — once aggressive buyers are exhausted (their orders filled at the high), there are no more buyers to push price up, and price naturally falls
4. **Symmetrical logic for longs** — aggressive sellers in lower wicks get absorbed by limit buyers

This is NOT a prediction of direction — it's a detection of exhausted flow that creates a temporary imbalance favoring the opposite direction.
