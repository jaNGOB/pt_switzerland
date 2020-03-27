import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as ts 
import statsmodels.api as sm
from numpy import log, polyfit, sqrt, std, subtract
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pyfinance.ols import PandasRollingOLS

#-----------------------------------------------------------------------

entry = 2
exit = 0

p_mavg = 60
p_stdev = 15
p_rsi = 14
rsi_thres = 0

#------------------------------------------------------------------------

def find_cointegrated_pairs(dataframe, critial_level = 0.02):
    n = dataframe.shape[1] # the length of dateframe
    pvalue_matrix = np.ones((n, n)) # initialize the matrix of p
    keys = dataframe.keys() # get the column names
    pairs = [] # initilize the list for cointegration
    for i in range(n):
        for j in range(i+1, n): # for j bigger than i
            stock1 = dataframe[keys[i]] # obtain the price of two contract
            stock2 = dataframe[keys[j]]
            result = ts.coint(stock1, stock2) # get conintegration
            pvalue = result[1] # get the pvalue
            pvalue_matrix[i, j] = pvalue
            if pvalue < critial_level: # if p-value less than the critical level
                pairs.append((keys[i], keys[j], pvalue)) # record the contract with that p-value
    return pairs


def backtest(temp, asset1, asset2):
    
    long = False
    short = False
    
    trades = 0
    
    long_price = None
    short_price = None
    
    last_price1 = None
    last_price2 = None

    ret = []
    
    for index, row in temp.iterrows():
        
        if row['longsignal'] and not long:
            
            #print('going long: X {} Y {}'.format(row[asset1], row[asset2]))
            
            trades += 1
            short = False
            long = True
            
            long_price = row[asset1]
            short_price = row[asset2]
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]
            
            ret.append(0)
            
        elif row['shortsignal'] and not short :
                        
            #print('going short: X {} Y {}'.format(row[asset1], row[asset2]))
            
            trades += 1
            long = False
            short = True
            
            long_price = row[asset2]
            short_price = row[asset1]
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]
            
            ret.append(0)
            
            
        elif long and row['closelong']:
            
            #print('closing long: X {} Y {}'.format(row[asset1], row[asset2]))
            
            short = False
            long = False
            
            long_ret = (row[asset1] / last_price1) - 1
            short_ret = (row[asset2] / -last_price2) + 1
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]
        
            ret.append(long_ret+short_ret)            
            
            
        elif short and row['closeshort']:
            
            #print('closing short: X {} Y {}'.format(row[asset1], row[asset2]))
            
            short = False
            long = False
            
            long_ret = (row[asset2] / last_price2) - 1
            short_ret = (row[asset1] / -last_price1) + 1
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]
        
            ret.append(long_ret+short_ret)            
            
            
        elif long:
            
            long_ret = (row[asset1] / last_price1) - 1
            short_ret = (row[asset2] / -last_price2) + 1
            
            long_price = row[asset1]
            short_price = row[asset2]
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]            
            
            ret.append(long_ret+short_ret)
            
        elif short:
            
            long_ret = (row[asset2] / last_price2) - 1
            short_ret = (row[asset1] / -last_price1) + 1
            
            long_price = row[asset2]
            short_price = row[asset1]
           
            last_price1 = row[asset1]
            last_price2 = row[asset2]
        
            ret.append(long_ret+short_ret)
            
        else:
            last_price1 = row[asset1]
            last_price2 = row[asset2]
            ret.append(0)

    temp['returns'] = ret
    temp['cum_returns'] = temp.returns.cumsum() + 1
    
    print(asset1, asset2, ': ', trades, 'trades were made')
    return temp


def backtest_with_cost(temp, asset1, asset2):
    
    long = False
    short = False
    
    trades = 0
    
    long_price = None
    short_price = None
    
    last_price1 = None
    last_price2 = None

    ret = []
    
    for index, row in temp.iterrows():
        
        if row['longsignal'] and not long:
            
            #print('going long: X {} Y {}'.format(row[asset1], row[asset2]))
            
            trades += 1
            short = False
            long = True
            
            long_price = row[asset1]
            short_price = row[asset2]
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]
            
            ret.append(-0.0084)
            
        elif row['shortsignal'] and not short :
                        
            #print('going short: X {} Y {}'.format(row[asset1], row[asset2]))
            
            trades += 1
            long = False
            short = True
            
            long_price = row[asset2]
            short_price = row[asset1]
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]
            
            ret.append(-0.0084)
            
            
        elif long and row['closelong']:
            
            #print('closing long: X {} Y {}'.format(row[asset1], row[asset2]))
            
            short = False
            long = False
            
            long_ret = (row[asset1] / last_price1) - 1
            short_ret = (row[asset2] / -last_price2) + 1
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]
        
            ret.append(long_ret+short_ret-0.0084)            
            
            
        elif short and row['closeshort']:
            
            #print('closing short: X {} Y {}'.format(row[asset1], row[asset2]))
            
            short = False
            long = False
            
            long_ret = (row[asset2] / last_price2) - 1
            short_ret = (row[asset1] / -last_price1) + 1
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]
        
            ret.append(long_ret+short_ret-0.0084)            
            
            
        elif long:
            
            long_ret = (row[asset1] / last_price1) - 1
            short_ret = (row[asset2] / -last_price2) + 1
            
            long_price = row[asset1]
            short_price = row[asset2]
            
            last_price1 = row[asset1]
            last_price2 = row[asset2]            
            
            ret.append(long_ret+short_ret)
            
        elif short:
            
            long_ret = (row[asset2] / last_price2) - 1
            short_ret = (row[asset1] / -last_price1) + 1
            
            long_price = row[asset2]
            short_price = row[asset1]
           
            last_price1 = row[asset1]
            last_price2 = row[asset2]
        
            ret.append(long_ret+short_ret)
            
        else:
            last_price1 = row[asset1]
            last_price2 = row[asset2]
            ret.append(0)

    temp['returns'] = ret
    temp['cum_returns'] = temp.returns.cumsum() + 1
    
    print(asset1, asset2, ': ', trades, 'trades were made')
    return temp


def pt_signal(df, stock1, stock2, entry, exit, p_mavg, p_stdev, p_rsi, rsi_thres):
    temp = df.copy()
    
    ret_df = pd.DataFrame()
    ret_df[stock1] = df[stock1]
    ret_df[stock2] = df[stock2]
    
    ratio = temp[stock1] / temp[stock2]
    mavg = ta.EMA(ratio, timeperiod=p_mavg)
    stdev = ta.STDDEV(ratio, timeperiod=p_stdev, nbdev=1)
    
    ret_df['rsi'] = ta.RSI(ratio, timeperiod=p_rsi)
    
    ret_df['ratio'] = ratio
    
    #ret_df['zscore'] = (ratio - mavg) / stdev
    
    ret_df['upper_entry'] = mavg + stdev * entry
    ret_df['lower_entry'] = mavg - stdev * entry
    ret_df['upper_exit'] = mavg + stdev * exit
    ret_df['lower_exit'] = mavg - stdev * exit
    
    ret_df['longsignal'] = (ret_df['ratio'] <= ret_df['lower_entry']) #& (ret_df['rsi'] <= (50-rsi_thres))
    ret_df['shortsignal'] = (ret_df['ratio'] >= ret_df['upper_entry']) #& (ret_df['rsi'] >= (50+rsi_thres))
    ret_df['closelong'] = (ret_df['ratio'] >= ret_df['lower_exit']) 
    ret_df['closeshort'] = (ret_df['ratio'] <= ret_df['upper_exit']) 
    
    return ret_df

def hr_signal(df, stock1, stock2):
    window = 20
    entry = 2
    exit = 0
    
    temp = pd.DataFrame()
    temp[stock1] = df[stock1]
    temp[stock2] = df[stock2]
    
    model = PandasRollingOLS(y=temp[stock1], x=temp[stock2], window=window)

    temp['beta'] = model.beta
    temp = temp.dropna()
    temp['spread'] = temp[stock1] - temp.beta*temp[stock2]
    temp['zscore'] = (temp['spread'] - np.mean(temp['spread']))/np.std(temp['spread'])
    
    temp['longsignal'] = (temp['zscore'] <= -entry) & (temp['zscore'] < -exit)
    temp['shortsignal'] = (temp['zscore'] >= entry) & (temp['zscore'] > exit)  
    temp['closelong'] = (temp['zscore'] >= -exit)
    temp['closeshort'] = (temp['zscore'] <= exit)
    
    return temp    
    

def plot_signal(df):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(df.ratio, color='black')
    ax.plot(df.upper_entry, color='g')
    ax.plot(df.lower_entry, color='g')
    ax.plot(df.upper_exit, color='r', linestyle='--')
    ax.plot(df.lower_exit, color='r', linestyle='--')
    
    plt.show()    

"""
df = pd.read_csv('SMI.csv', index_col='Date', parse_dates=True)
df = df.drop(columns="ALC.SW")
df = df.ffill()    
    
current_quarter = df.index[0]

ret = pd.DataFrame(df['2007'])

def quarterly_reb(df):
    current_quarter = df.index[0]
    for index, row in df['2007'].iterrows():
        if index < current_quarter and current_quarter.isoweekday() in range(1, 6):
            pass
        else:
            current_quarter = (index + pd.offsets.QuarterEnd() - pd.DateOffset(1))
            if current_quarter.isoweekday() in range(6, 7):
                current_quarter = (current_quarter - pd.DateOffset(1))

        if index == current_quarter:
            print(current_quarter)
            next_quarter = ((index + pd.DateOffset(5))+ pd.offsets.QuarterEnd())
            pairs = find_cointegrated_pairs(df[:index])
            print('found the following pairs: ', pairs)
            for n in range(len(pairs)):
                stock1 = pairs[n][0]
                stock2 = pairs[n][1]

                print(stock1, stock2)
                name = str(index) + stock1 + "-" + stock2 

                #signal = pt_signal(df, stock1, stock2, entry, exit, p_mavg, p_stdev, p_rsi, rsi_thres)
                signal = hr_signal(df, stock1, stock2)

                tmp = backtest(signal[index:next_quarter], stock1, stock2)

                ret[name] = tmp.cum_returns

            print('----------------------------')
            
def monthly_reb(df):
    current_month = df.index[0]

    for index, row in df['2017'].iterrows():
        if index < current_month and current_month.isoweekday() in range(1, 6):
            pass
        else:
            current_month = (index + pd.offsets.MonthEnd() - pd.DateOffset(1))
            if current_month.isoweekday() in range(6, 7):
                current_month = (current_month - pd.DateOffset(1))

        if index == current_month:
            print(current_month)
            next_quarter = ((index + pd.DateOffset(5))+ pd.offsets.QuarterEnd())
            pairs = find_cointegrated_pairs(df[:index])
            print('found the following pairs: ', pairs)
            for n in range(len(pairs)):
                stock1 = pairs[n][0]
                stock2 = pairs[n][1]

                print(stock1, stock2)
                name = str(index) + stock1 + "-" + stock2 

                signal = pt_signal(df, stock1, stock2, entry, exit, p_mavg, p_stdev, p_rsi, rsi_thres)

                tmp = backtest(signal[index:next_quarter], stock1, stock2)

                ret[name] = tmp.cum_returns

            print('----------------------------')     
            
#quarterly_reb(df)
#print(ret)
ret.to_csv('returns.csv')

"""