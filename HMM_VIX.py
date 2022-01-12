# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:02:00 2020

@author: Country Dragon
"""

import pandas as pd
import numpy as np
from random import randint
from sklearn.preprocessing import scale
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import MonthLocator,YearLocator
from hmmlearn.hmm import GaussianHMM, GMMHMM
from talib import BBANDS, ATR, MA

df = pd.read_csv('VIX_Daily.csv')

quotes = []

df['B_Bands_Upper'], df['B_Bands_Mid'], df['B_Bands_Lower'] = BBANDS(df['Close'],5,2,2,0)
df['ATR'] = ATR(df['High'],df['Low'],df['Close'],14)
df['MA_High'] = MA(df['High'],10,0)

df_train = df[:7075]
df_test = df[7075:].reset_index(drop = True)

for row_set in range(0,100000):
    if row_set%2000==0: print(row_set)
    row_quant = randint(10,30) #sequence 10 to 30 trading days as events
    row_start = randint(20, len(df_train)-row_quant)
    subset = df_train.iloc[row_start:row_start+row_quant]
    
    close_Date = max(subset['Date'])
    if row_set%2000==0: print(close_Date)
    
    log_return_rate = np.log(subset['Close']) - np.log(subset['Close'].shift(1))
    log_diff_high_low = np.log(subset['High']) - np.log(subset['Low'])
    log_atr = np.log(subset['ATR']) - np.log(subset['ATR'].shift(1))
    log_bbwidth = np.log(subset['B_Bands_Upper'] - subset['B_Bands_Lower']) - np.log(subset['B_Bands_Upper'].shift(1) - subset['B_Bands_Lower'].shift(1))
    log_MA_diff = np.log(subset['MA_High']) - np.log(subset['Close'])
    log_op_gap = np.log(subset['Open'].shift(-1)) - np.log(subset['Close'])
    log_hi_gap = np.log(subset['Open'].shift(-1)) - np.log(subset['High'])
    log_op_close = np.log(subset['Close']) - np.log(subset['Open'])

    quotes.append(pd.DataFrame({'Sequence_ID':[row_set]*len(subset),
                                'close_date':[close_Date]*len(subset),
                                'log_return_rate':log_return_rate,
                                'log_bbwidth':log_bbwidth,
                                'log_diff_high_low': log_diff_high_low,
                                'log_atr':log_atr,
                                'log_MA_diff':log_MA_diff,
                                'log_op_gap': log_op_gap,
                                'log_hi_gap': log_hi_gap,
                                'log_op_close': log_op_close}))

quotes_df = pd.concat(quotes)
print(quotes_df.shape)

quotes_df.count()


#drop missing values
quotes_df = quotes_df.dropna(how='any')
print(quotes_df.shape)
print(quotes_df.tail(10))


log_return_rate = np.array(quotes_df['log_return_rate'].values)
log_bbwidth = np.array(quotes_df['log_bbwidth'].values)
log_atr = np.array(quotes_df['log_atr'].values)
log_diff_high_low = np.array(quotes_df['log_diff_high_low'].values)
log_MA_diff = np.array(quotes_df['log_MA_diff'].values)
log_op_gap = np.array(quotes_df['log_op_gap'].values)
log_hi_gap = np.array(quotes_df['log_hi_gap'].values)
log_op_close = np.array(quotes_df['log_op_close'].values)


X = np.column_stack([log_return_rate,
                     log_diff_high_low,
                     log_bbwidth,
                     log_atr,
                     log_MA_diff,
                     log_op_gap,
                     log_hi_gap]) 


model = GaussianHMM(n_components = 5, covariance_type = 'diag')

model.fit(X)

model.score(X)

#BIC Evaluation
def bic_general(likelihood_fn, k, X):

    bic = np.log(len(X))*k - 2*likelihood_fn(X)
    return bic

def bic_hmmlearn(X):
    lowest_bic = np.infty
    bic = []
    n_states_range = range(1,20)
    for n_components in n_states_range:
        hmm_curr = GaussianHMM(n_components=n_components, covariance_type='diag')
        hmm_curr.fit(X)

        # Calculate number of free parameters
        # free_parameters = for_means + for_covars + for_transmat + for_startprob
        # for_means & for_covars = n_features*n_components
        n_features = hmm_curr.n_features
        free_parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)

        bic_curr = bic_general(hmm_curr.score, free_parameters, X)
        bic.append(bic_curr)
        if bic_curr < lowest_bic:
            lowest_bic = bic_curr
        best_hmm = hmm_curr

    return (best_hmm, bic)

best_hmm, bic = bic_hmmlearn(X)



#PREDICTION


#Import of new data
import yfinance as yf
import pandas as pd

nq = yf.Ticker('^VIX')

hist = nq.history(start = '2019-01-01',actions = False)
hist.to_csv('VIX_Daily_2020.csv')

prediction_set = []
df_custom = pd.read_csv('VIX_Daily_2020.csv')


df_custom['B_Bands_Upper'], df_custom['B_Bands_Mid'], df_custom['B_Bands_Lower'] = BBANDS(df_custom['Close'],5,2,2,0)
df_custom['ATR'] = ATR(df_custom['High'],df_custom['Low'],df_custom['Close'],14)
df_custom['MA_High'] = MA(df_custom['High'],10,0)

#Prediction data input
df_predict = df_custom

for i in range(20,len(df_predict)-1):
    log_return_rate = np.log(df_predict.loc[i,'Close']) - np.log(df_predict.loc[i-1,'Close'])
    log_diff_high_low = np.log(df_predict.loc[i,'High']) - np.log(df_predict.loc[i,'Low'])
    log_bbwidth = np.log(df_predict.loc[i,'B_Bands_Upper'] - df_predict.loc[i,'B_Bands_Lower']) - np.log(df_predict.loc[i-1,'B_Bands_Upper'] - df_predict.loc[i-1,'B_Bands_Lower'])
    log_atr =  np.log(df_predict.loc[i,'ATR']) - np.log(df_predict.loc[i-1,'ATR'])
    log_MA_diff = np.log(df_predict.loc[i,'MA_High']) - np.log(df_predict.loc[i,'Close'])
    log_op_gap = np.log(df_predict.loc[i+1,'Open']) - np.log(df_predict.loc[i,'Close'])
    log_hi_gap = np.log(df_predict.loc[i+1,'Open']) - np.log(df_predict.loc[i,'Close'])
    log_op_close = np.log(df_predict.loc[i,'Close']) - np.log(df_predict.loc[i,'Open'])
    
    close_price = df_predict.loc[i,'Close']
    
    prediction_set.append(pd.DataFrame({'Date': df_predict.loc[i,'Date'],
                                'log_return_rate':log_return_rate,
                                'log_diff_high_low':log_diff_high_low,
                                'log_bbwidth':log_bbwidth,
                                'log_atr':log_atr,
                                'log_MA_diff':log_MA_diff,
                                'log_op_gap':log_op_gap,
                                'log_hi_gap':log_hi_gap,
                                'log_op_close':log_op_close,
                                'close_price':close_price}, index =[0]))    

prediction_set_df = pd.concat(prediction_set)

prediction_set_df = prediction_set_df.dropna(how='any')
print(prediction_set_df.shape)
print(prediction_set_df.tail(10))

prediction_log_return_rate = np.array(prediction_set_df['log_return_rate'].values)
prediction_log_bbwidth = np.array(prediction_set_df['log_bbwidth'].values)
prediction_log_atr = np.array(prediction_set_df['log_atr'].values)
prediction_log_diff_high_low = np.array(prediction_set_df['log_diff_high_low'].values)
prediction_log_MA_diff = np.array(prediction_set_df['log_MA_diff'].values)
prediction_log_op_gap = np.array(prediction_set_df['log_op_gap'].values)
prediction_log_hi_gap = np.array(prediction_set_df['log_hi_gap'].values)
prediction_log_op_close = np.array(prediction_set_df['log_op_close'].values)

P = np.column_stack([prediction_log_return_rate,
                     prediction_log_diff_high_low,
                     prediction_log_bbwidth,
                     prediction_log_atr,
                     prediction_log_MA_diff,
                     prediction_log_op_gap,
                     prediction_log_hi_gap])
 
prediction_set_df['Hidden State'] = model.predict(P)    

colors = {0:'red', 1:'orange', 2:'yellow', 3:'green', 4: 'aqua',5:'blue',6:'purple',7:'black'}

fig, ax = plt.subplots()
ax.scatter(prediction_set_df['Date'],prediction_set_df['close_price'], c = prediction_set_df['Hidden State'].map(colors))
plt.show()

prediction_set_df_hs1 = prediction_set_df[prediction_set_df['Hidden State']==1]

model.transmat_
model.score(X)



import pickle

#Save Model
with open("VIX_5S_Model3.pkl", "wb") as file: pickle.dump(model, file)

#Load Model
model = pickle.load( open( "VIX_5S_Model2.pkl", "rb" ) )