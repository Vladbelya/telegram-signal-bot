import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from itertools import product

SYMBOL="ETHUSDT"; SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_FILE=os.path.join(SCRIPT_DIR,"eth_2025_data.csv"); INITIAL_CAPITAL=10000.0
TIMEFRAME="1h"; TF_LABEL="1H"; TF_MINUTES=60

def load_data(filepath=DATA_FILE):
    if not os.path.exists(filepath): print(f"[ERROR] {filepath} not found."); return None
    print(f"[INFO] Loading 1m data from {filepath}...")
    df=pd.read_csv(filepath); tc='ts' if 'ts' in df.columns else ('timestamp' if 'timestamp' in df.columns else df.columns[0])
    df[tc]=pd.to_datetime(df[tc]); df=df[(df[tc]>='2025-01-01')&(df[tc]<='2025-12-31')]
    if 'sell_vol' not in df.columns: df['sell_vol']=df['vol']-df['taker_buy_vol']
    df.set_index(tc,inplace=True); df.sort_index(inplace=True); return df

def prepare_tf_data(df_1m):
    print(f"[INFO] Resampling to {TF_LABEL}...")
    df_tf=df_1m.resample(TIMEFRAME).agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    df_1m['tf_ts']=df_1m.index.floor(TIMEFRAME)
    m=pd.merge(df_1m.reset_index(),df_tf[['open','close']].reset_index(),left_on='tf_ts',right_on='ts',suffixes=('','_tf'))
    if 'ts_tf' in m.columns: m.drop(columns=['ts_tf'],inplace=True)
    m['wt']=m[['open_tf','close_tf']].max(axis=1); m['wb']=m[['open_tf','close_tf']].min(axis=1)
    mu=m['close']>=m['wt']; ml=m['close']<=m['wb']
    df_tf['up_wick_buy']=m[mu].groupby('tf_ts')['taker_buy_vol'].sum()
    df_tf['up_wick_sell']=m[mu].groupby('tf_ts')['sell_vol'].sum()
    df_tf['lo_wick_buy']=m[ml].groupby('tf_ts')['taker_buy_vol'].sum()
    df_tf['lo_wick_sell']=m[ml].groupby('tf_ts')['sell_vol'].sum()
    df_tf.fillna(0,inplace=True)
    df_tf['up_wick_total']=df_tf['up_wick_buy']+df_tf['up_wick_sell']
    df_tf['lo_wick_total']=df_tf['lo_wick_buy']+df_tf['lo_wick_sell']
    aw=pd.concat([df_tf[df_tf['up_wick_total']>0]['up_wick_total'],df_tf[df_tf['lo_wick_total']>0]['lo_wick_total']])
    pct={p:np.percentile(aw,100-p) for p in range(1,11)}
    print(f"[INFO] {SYMBOL} {TF_LABEL} Thresholds:")
    for p,v in pct.items(): print(f"  Top {p}%: {v:.2f}")
    return df_tf,pct

def backtest(df_tf_o,df_1m,tp_pct,imb,rr,sl_buf,pct):
    vt=pct[tp_pct]; df=df_tf_o.copy()
    sc=(df['up_wick_total']>=vt)&(df['up_wick_buy']>=df['up_wick_sell']*imb)&(df['up_wick_sell']>0)
    lc=(df['lo_wick_total']>=vt)&(df['lo_wick_sell']>=df['lo_wick_buy']*imb)&(df['lo_wick_buy']>0)
    df['signal']=0; df.loc[sc,'signal']=-1; df.loc[lc,'signal']=1; df.loc[sc&lc,'signal']=0
    if df[df['signal']!=0].empty:
        return {'Total Trades':0,'Winrate %':0,'Net Profit ($)':0,'Max Drawdown (%)':0,'Profit Factor':0,'trades':[],'equity_curve':[INITIAL_CAPITAL]}
    cap=INITIAL_CAPITAL; eq=[cap]; trades=[]
    t1=df_1m.index.values; h1=df_1m['high'].values; l1=df_1m['low'].values; ttf=df.index.values
    i=0
    while i<len(df)-1:
        s=df['signal'].iloc[i]
        if s!=0:
            b=df.iloc[i]; et=df.index[i]; ex=et+pd.Timedelta(minutes=TF_MINUTES); ep=b['close']
            if s==-1:
                pt=-1; sl=b['high']*(1+sl_buf); d=abs(sl-ep)
                if d<0.1: sl=ep+(ep*0.001); d=abs(sl-ep)
                sz=(cap*0.01)/d; tp=ep-(d*rr)
            else:
                pt=1; sl=b['low']*(1-sl_buf); d=abs(ep-sl)
                if d<0.1: sl=ep-(ep*0.001); d=abs(ep-sl)
                sz=(cap*0.01)/d; tp=ep+(d*rr)
            si=np.searchsorted(t1,np.datetime64(ex)); cl=False
            for mi in range(si,len(t1)):
                h=h1[mi]; l=l1[mi]; hs=False; ht=False
                if pt==-1:
                    if h>=sl: hs=True
                    if l<=tp: ht=True
                else:
                    if l<=sl: hs=True
                    if h>=tp: ht=True
                if hs and ht: xp=sl; cl=True
                elif hs: xp=sl; cl=True
                elif ht: xp=tp; cl=True
                if cl:
                    g=(xp-ep)*sz if pt==1 else (ep-xp)*sz
                    c=(ep*sz*0.0004)+(xp*sz*0.0004); sp=(ep*sz*0.0001)+(xp*sz*0.0001)
                    n=g-c-sp; cap+=n; eq.append(cap)
                    trades.append({'Net PnL':n,'Result':'Win' if n>0 else 'Loss'})
                    ni=np.searchsorted(ttf,t1[mi]); i=max(i,ni); break
            if not cl: break
        i+=1
    tt=len(trades)
    if tt==0: return {'Total Trades':0,'Winrate %':0,'Net Profit ($)':0,'Max Drawdown (%)':0,'Profit Factor':0,'trades':trades,'equity_curve':eq}
    w=sum(1 for t in trades if t['Result']=='Win'); wr=(w/tt)*100; np_=cap-INITIAL_CAPITAL
    es=pd.Series(eq); mdd=((es.cummax()-es)/es.cummax()).max()*100
    gp=sum(t['Net PnL'] for t in trades if t['Net PnL']>0); gl=sum(t['Net PnL'] for t in trades if t['Net PnL']<0)
    pf=gp/abs(gl) if gl!=0 else np.inf
    return {'Top % Volume':tp_pct,'Imbalance Ratio':imb,'R:R':rr,'Total Trades':tt,'Winrate %':round(wr,2),'Net Profit ($)':round(np_,2),'Max Drawdown (%)':round(mdd,2),'Profit Factor':round(pf,2),'trades':trades,'equity_curve':eq}

def run_grid_search(df_tf,df_1m,pct):
    tp_p=list(range(1,11)); imbs=[1.6,2.0,3.0,4.0]; rrs=[1.0,1.5,2.0,2.5,3.0]; slbs=[0.0005,0.001,0.0015]
    res=[]; total=len(tp_p)*len(imbs)*len(rrs)*len(slbs)
    print(f"\n[INFO] {SYMBOL} {TF_LABEL} Grid Search: {total} combos...")
    c=0
    for t,im,r,sb in product(tp_p,imbs,rrs,slbs):
        c+=1; rs=backtest(df_tf,df_1m,t,im,r,sb,pct)
        if rs: rs['SL Buffer %']=sb*100; res.append(rs)
        if c%20==0: print(f"  Progress: {c}/{total}...")
    df_r=pd.DataFrame([{k:v for k,v in r.items() if k not in['trades','equity_curve']} for r in res])
    df_r.sort_values('Net Profit ($)',ascending=False,inplace=True)
    print(f"\n[OK] {SYMBOL} {TF_LABEL} Complete ({len(df_r)} results). Top 15:")
    print(df_r.head(15).to_string(index=False))
    df_r.to_csv(f"eth_anomaly_{TF_LABEL.lower()}_grid_results.csv",index=False)
    best=max(res,key=lambda x:x['Net Profit ($)'])
    plt.figure(figsize=(12,6)); plt.plot(best['equity_curve'])
    plt.title(f"{SYMBOL} Best {TF_LABEL}"); plt.ylabel("Capital ($)"); plt.grid(True)
    plt.savefig(f"eth_best_{TF_LABEL.lower()}_anomaly_equity.png")

if __name__=="__main__":
    df_1m=load_data()
    if df_1m is not None: df_tf,p=prepare_tf_data(df_1m); run_grid_search(df_tf,df_1m,p)
