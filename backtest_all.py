import statsmodels.tsa.stattools as ts 
from numpy import log, polyfit, sqrt, std, subtract
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dateutil.relativedelta import *

from backtester import backtest as bt
from backtester import backtest_with_cost as btwc


def create_pairs(tickers):
        result = []
        for p1 in range(len(tickers)):
                for p2 in range(p1+1,len(tickers)):
                        result.append([tickers[p1],tickers[p2]])
        return result
    
def get_closing(temp, name):
    temp.DATE = pd.to_datetime(temp['DATE'].str[:10], dayfirst=True)
    temp = temp.set_index(temp.DATE)
    
    temp.columns = ['Date', 'Symbol', 'Interval', 'Open', 'High', 'Low', name, 'Volume']
    temp = temp[name]
    return temp

    
def find_ssd(df, pairs):
    ssd = {}
    for p in pairs:
        pair = p[0]+','+p[1]
        A = df[p[0]]
        B = df[p[1]]
        s = np.sum((A-B)**2)
        
        if s == 0.0:
            #s = 999.9 
            continue
        
        if len(A.dropna()) < 248 or len(B.dropna()) < 248:
            #s = 999.9
            continue
            
        ssd[pair] = s
        
    return ssd


def find_tradeble_pairs(ssd, amount):
    sorted_lst = sorted(ssd, key=ssd.get)
    pair_lst = []
    if len(sorted_lst) < 1:
        return pair_lst
    for n in range(amount):
        pair_lst.append(sorted_lst[n].split(","))
    
    return pair_lst


def find_entry_exit(spread, n):
    mean = np.mean(spread)
    stdev = np.std(spread)
    upper = mean+(n*stdev)
    lower = mean-(n*stdev)
    
    return mean, upper, lower

def create_signal(df, pair, m, u, l):
    spread = df[pair[0]] - df[pair[1]]
    previous_spread = spread.shift(1)
    bt_df = pd.DataFrame()
    
    bt_df[pair[0]] = df[pair[0]]
    bt_df[pair[1]] = df[pair[1]]
    bt_df['shortsignal'] = ((spread < u) & (previous_spread > u))
    bt_df['longsignal'] = ((spread > l) & (previous_spread < l))
    bt_df['closelong'] = np.where(spread >= m, 1, 0)
    bt_df['closeshort'] = np.where(spread <= m, 1, 0)
    
    return bt_df


def trading(df_formation, df_trading, amount, n):
    returns = pd.DataFrame()
    
    ssd = find_ssd(df_formation, pairs)
    selected = find_tradeble_pairs(ssd, amount)
    
    for pair in selected:
        bt_name = pair[0]+pair[1]
        
        spread = df_formation[pair[0]] - df_formation[pair[1]]
        m, u, l = find_entry_exit(spread, n) 
        
        bt_df = create_signal(df_trading, pair, m, u, l)
        ret = bt(bt_df, pair[0], pair[1])
        
        returns[bt_name] = ret.returns
        
    returns['mean'] = returns.mean(axis=1)
    returns = returns.set_index(df_trading.index)
    return returns['mean']
    
    

def backtest(df, amount, n):
    overall_ret = pd.DataFrame()
    
    start = pd.Timestamp('2000-01')
    mid = start + relativedelta(months=+12)
    end = start + relativedelta(months=+18)
    
    df_formation = (1 + df[start:mid].pct_change()).cumprod()
    df_trading = (1 + df[mid:end].pct_change()).cumprod()
    
    returns = trading(df_formation, df_trading, amount, n)
    overall_ret = returns
    
    #print(start, mid, end)

    for period in range(1, 38):
        start = start + relativedelta(months=+6)
        mid = start + relativedelta(months=+12)
        end = start + relativedelta(months=+18)
        
        df_formation = (1 + df[start:mid].pct_change()).cumprod()
        df_trading = (1 + df[mid:end].pct_change()).cumprod()

        returns = trading(df_formation, df_trading, amount, n)
        overall_ret = overall_ret.append(returns)

        #print(start, mid, end)
        
    return overall_ret


amounts = [2, 5, 10]
n = [0.5, 1.0 , 1.5, 2.0]
sectors = ['hc', 'cg', 'fin', 'ind', 'all']


for sector in sectors:
    UBS = pd.read_csv('C:/Users/jango/code/research_env/BTHE/Datalink/csv/UBSG.csv', delimiter=";")
    UBS.DATE = pd.to_datetime(UBS['DATE'].str[:10], dayfirst=True)
    df = pd.DataFrame(UBS.CLOSE).set_index(UBS.DATE)
    df.columns = ['UBS']
    
    spreadsheet = pd.read_excel('C:/Users/jango/Documents/BFH/BTHE/Sectors.xlsx', sheet_name=None)
    tickers = spreadsheet[sector].Ticker.tolist()

    pairs = create_pairs(tickers)
    
    for ticker in tickers:
        path = 'C:/Users/jango/code/research_env/BTHE/Datalink/csv/'+ticker+'.csv'
        df = df.join(get_closing(pd.read_csv(path, delimiter=";"), ticker)).ffill()
    df = df.drop(columns=['UBS'])  
    
    save_name = 'results_'+sector+'.csv'
    master_lst = {}
    master_cum = {}

    for number in tqdm(n):
        for amount in amounts:
            name = '{}ssd{}std'.format(amount, number)
            ret = backtest(df, amount, number)
            master_lst[name] = ret
            master_cum[name] = (1 + ret).cumprod()
    master_df = pd.DataFrame(master_cum)
    master_df.to_csv(save_name)
    print(save_name, 'finished')