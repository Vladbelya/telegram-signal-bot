import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

SYMBOL = "ETHUSDT"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "eth_2025_data.csv")
INITIAL_CAPITAL = 10000.0
TIMEFRAME = "15min"
TF_LABEL = "15M"
TF_MINUTES = 15

def load_data(filepath=DATA_FILE):
    if not os.path.exists(filepath):
        print(f"[ERROR] Data file {filepath} not found.")
        return None
    print(f"[INFO] Loading 1m data from {filepath}...")
    df_1m = pd.read_csv(filepath)
    time_col = 'ts' if 'ts' in df_1m.columns else ('timestamp' if 'timestamp' in df_1m.columns else df_1m.columns[0])
    df_1m[time_col] = pd.to_datetime(df_1m[time_col])
    df_1m = df_1m[(df_1m[time_col] >= '2025-01-01') & (df_1m[time_col] <= '2025-12-31')]
    if 'sell_vol' not in df_1m.columns:
        df_1m['sell_vol'] = df_1m['vol'] - df_1m['taker_buy_vol']
    df_1m.set_index(time_col, inplace=True)
    df_1m.sort_index(inplace=True)
    return df_1m

def prepare_tf_data(df_1m):
    print(f"[INFO] Resampling 1m to {TF_LABEL}...")
    df_tf = df_1m.resample(TIMEFRAME).agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    df_1m['tf_ts'] = df_1m.index.floor(TIMEFRAME)
    merged = pd.merge(df_1m.reset_index(), df_tf[['open','close']].reset_index(), left_on='tf_ts', right_on='ts', suffixes=('','_tf'))
    if 'ts_tf' in merged.columns: merged.drop(columns=['ts_tf'], inplace=True)
    merged['wick_top'] = merged[['open_tf','close_tf']].max(axis=1)
    merged['wick_bot'] = merged[['open_tf','close_tf']].min(axis=1)
    mask_up = merged['close'] >= merged['wick_top']
    upper_buys = merged[mask_up].groupby('tf_ts')['taker_buy_vol'].sum()
    upper_sells = merged[mask_up].groupby('tf_ts')['sell_vol'].sum()
    mask_lo = merged['close'] <= merged['wick_bot']
    lower_buys = merged[mask_lo].groupby('tf_ts')['taker_buy_vol'].sum()
    lower_sells = merged[mask_lo].groupby('tf_ts')['sell_vol'].sum()
    df_tf['up_wick_buy'] = upper_buys; df_tf['up_wick_sell'] = upper_sells
    df_tf['lo_wick_buy'] = lower_buys; df_tf['lo_wick_sell'] = lower_sells
    df_tf.fillna(0, inplace=True)
    df_tf['up_wick_total'] = df_tf['up_wick_buy'] + df_tf['up_wick_sell']
    df_tf['lo_wick_total'] = df_tf['lo_wick_buy'] + df_tf['lo_wick_sell']
    all_wv = pd.concat([df_tf[df_tf['up_wick_total']>0]['up_wick_total'], df_tf[df_tf['lo_wick_total']>0]['lo_wick_total']])
    percentiles = {p: np.percentile(all_wv, 100-p) for p in range(1,11)}
    print(f"[INFO] {SYMBOL} {TF_LABEL} Thresholds:")
    for p, val in percentiles.items(): print(f"  Top {p}%: {val:.2f}")
    return df_tf, percentiles

def backtest_strategy(df_tf_orig, df_1m, top_percent, imbalance_ratio, rr_ratio, sl_buffer_pct, percentiles_dict):
    volume_threshold = percentiles_dict[top_percent]
    df_tf = df_tf_orig.copy()
    short_cond = (df_tf['up_wick_total']>=volume_threshold)&(df_tf['up_wick_buy']>=df_tf['up_wick_sell']*imbalance_ratio)&(df_tf['up_wick_sell']>0)
    long_cond = (df_tf['lo_wick_total']>=volume_threshold)&(df_tf['lo_wick_sell']>=df_tf['lo_wick_buy']*imbalance_ratio)&(df_tf['lo_wick_buy']>0)
    df_tf['signal']=0; df_tf.loc[short_cond,'signal']=-1; df_tf.loc[long_cond,'signal']=1
    df_tf.loc[short_cond&long_cond,'signal']=0
    if df_tf[df_tf['signal']!=0].empty:
        return {'Total Trades':0,'Winrate %':0,'Net Profit ($)':0,'Max Drawdown (%)':0,'Profit Factor':0,'trades':[],'equity_curve':[INITIAL_CAPITAL]}
    capital=INITIAL_CAPITAL; equity_curve=[capital]; trades=[]
    ts_1m=df_1m.index.values; high_1m=df_1m['high'].values; low_1m=df_1m['low'].values
    ts_tf=df_tf.index.values
    idx=0
    while idx<len(df_tf)-1:
        sig=df_tf['signal'].iloc[idx]
        if sig!=0:
            bar=df_tf.iloc[idx]; entry_time=df_tf.index[idx]
            exec_time=entry_time+pd.Timedelta(minutes=TF_MINUTES); entry_price=bar['close']
            if sig==-1:
                pos=-1; sl=bar['high']*(1+sl_buffer_pct); d=abs(sl-entry_price)
                if d<0.1: sl=entry_price+(entry_price*0.001); d=abs(sl-entry_price)
                sz=(capital*0.01)/d; tp=entry_price-(d*rr_ratio)
            else:
                pos=1; sl=bar['low']*(1-sl_buffer_pct); d=abs(entry_price-sl)
                if d<0.1: sl=entry_price-(entry_price*0.001); d=abs(entry_price-sl)
                sz=(capital*0.01)/d; tp=entry_price+(d*rr_ratio)
            si=np.searchsorted(ts_1m,np.datetime64(exec_time)); closed=False
            for mi in range(si,len(ts_1m)):
                h=high_1m[mi]; l=low_1m[mi]; hsl=False; htp=False
                if pos==-1:
                    if h>=sl: hsl=True
                    if l<=tp: htp=True
                else:
                    if l<=sl: hsl=True
                    if h>=tp: htp=True
                if hsl and htp: ep=sl; et="SL"; closed=True
                elif hsl: ep=sl; et="SL"; closed=True
                elif htp: ep=tp; et="TP"; closed=True
                if closed:
                    gross=(ep-entry_price)*sz if pos==1 else (entry_price-ep)*sz
                    comm=(entry_price*sz*0.0004)+(ep*sz*0.0004)
                    slip=(entry_price*sz*0.0001)+(ep*sz*0.0001)
                    net=gross-comm-slip; capital+=net; equity_curve.append(capital)
                    trades.append({'Net PnL':net,'Result':'Win' if net>0 else 'Loss'})
                    ni=np.searchsorted(ts_tf,ts_1m[mi]); idx=max(idx,ni); break
            if not closed: break
        idx+=1
    tt=len(trades)
    if tt==0: return {'Total Trades':0,'Winrate %':0,'Net Profit ($)':0,'Max Drawdown (%)':0,'Profit Factor':0,'trades':trades,'equity_curve':equity_curve}
    w=sum(1 for t in trades if t['Result']=='Win'); wr=(w/tt)*100; np_=capital-INITIAL_CAPITAL
    eq=pd.Series(equity_curve); mdd=((eq.cummax()-eq)/eq.cummax()).max()*100
    gp=sum(t['Net PnL'] for t in trades if t['Net PnL']>0); gl=sum(t['Net PnL'] for t in trades if t['Net PnL']<0)
    pf=gp/abs(gl) if gl!=0 else np.inf
    return {'Top % Volume':top_percent,'Imbalance Ratio':imbalance_ratio,'R:R':rr_ratio,'Total Trades':tt,'Winrate %':round(wr,2),'Net Profit ($)':round(np_,2),'Max Drawdown (%)':round(mdd,2),'Profit Factor':round(pf,2),'trades':trades,'equity_curve':equity_curve}

def run_grid_search(df_tf, df_1m, pct):
    top_p=list(range(1,11)); imb=[1.6,2.0,3.0,4.0]; rr=[1.0,1.5,2.0,2.5,3.0]; slb=[0.0005,0.001,0.0015]
    results=[]; total=len(top_p)*len(imb)*len(rr)*len(slb)
    print(f"\n[INFO] {SYMBOL} {TF_LABEL} Grid Search: {total} combinations...")
    i=0
    for tp,im,r,sb in product(top_p,imb,rr,slb):
        i+=1; res=backtest_strategy(df_tf,df_1m,tp,im,r,sb,pct)
        if res: res['SL Buffer %']=sb*100; results.append(res)
        if i%20==0: print(f"  Progress: {i}/{total}...")
    df_r=pd.DataFrame([{k:v for k,v in r.items() if k not in['trades','equity_curve']} for r in results])
    df_r.sort_values('Net Profit ($)',ascending=False,inplace=True)
    print(f"\n[OK] {SYMBOL} {TF_LABEL} Complete ({len(df_r)} results). Top 15:")
    print(df_r.head(15).to_string(index=False))
    df_r.to_csv(f"eth_anomaly_{TF_LABEL.lower()}_grid_results.csv",index=False)
    best=max(results,key=lambda x:x['Net Profit ($)'])
    plt.figure(figsize=(12,6)); plt.plot(best['equity_curve'])
    plt.title(f"{SYMBOL} Best {TF_LABEL}"); plt.ylabel("Capital ($)"); plt.grid(True)
    plt.savefig(f"eth_best_{TF_LABEL.lower()}_anomaly_equity.png"); print(f"[INFO] Saved equity curve.")

if __name__=="__main__":
    df_1m=load_data()
    if df_1m is not None:
        df_tf,pct=prepare_tf_data(df_1m); run_grid_search(df_tf,df_1m,pct)
