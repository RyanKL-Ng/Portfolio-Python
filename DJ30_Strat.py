# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:12:14 2021

@author: Country Dragon
"""

#PART 1: DATA PULL
import numpy as np
import pandas as pd
import requests
import csv
import os

const = pd.read_csv('DJ30_constituents.csv')
headers = {
    'Content-Type': 'application/json'
}

token = os.environ.get('tiingo_tok')

for x in const.iloc[:,0]:
    url = "https://api.tiingo.com/tiingo/daily/"+x+"/prices?startDate=2010-01-01&token="+token
    requestResponse = requests.get(url, headers=headers)
    data = requestResponse.json()
    
    if data:    
        filename = "TIINGO_DATA_10Y/"+x+"_data_file.csv"
        # now we will open a file for writing
        data_file = open(filename, 'w',newline='')
          
        # create the csv writer object
        csv_writer = csv.writer(data_file)
          
        # Counter variable used for writing 
        # headers to the CSV file
        count = 0
          
        for x in data:
            if count == 0: 
                # Writing headers of CSV file
                header = x.keys()
                csv_writer.writerow(header)
                count += 1
          
            # Writing data of CSV file
            csv_writer.writerow(x.values())
          
        data_file.close()
    else:
        print(x+": No data")


#PART 2: FEATURE EXTRACTION & ANALYSIS
from tapy import Indicators
from finta import TA
import numpy as np
import pandas as pd
import talib
import datetime as dt


def process_data(symbol):    
    #2.1: FILTERING ONLY ADJUSTED DATA
    path ='TIINGO_DATA_10Y/'+symbol+'_data_file.csv'
    df = pd.read_csv(path)
    df = df.drop(['close','high','low','open','volume','divCash','splitFactor'],axis = 1)
    df.columns = ['date','close','high','low','open','volume']
    df['symbol'] = symbol
    
    #2.2: FORMATTING DATE AND GENERATING FRACTALS DATA
    df_bull = []
    
    df_bull = TA.WILLIAMS_FRACTAL(df,5)
    df = pd.concat([df, df_bull], axis=1, join="inner")
    
    df_datetime_date = []
    df_bullish_pts = []
    df_bearish_pts = []
    
    for index, row in df.iterrows():    
        reference = 0
        
        df_datetime_date.append(dt.datetime.strptime(row['date'][:10], '%Y-%m-%d'))
        
        for i in range(2,300):
            if index - i < len(df):
                ref = df.iloc[index-i]
            
            if ref['BullishFractal'] == 1:
                reference = df.iloc[index-i]['low']
                break
            
        df_bullish_pts.append(reference)
    
    for index, row in df.iterrows():    
        reference = 0
        
        for i in range(2,300):
            if index - i < len(df):
                ref = df.iloc[index-i]
            
            if ref['BearishFractal'] == 1:
                reference = df.iloc[index-i]['high']
                break
            
        df_bearish_pts.append(reference)
    
    df['low fractal pts'] = df_bullish_pts
    df['high fractal pts'] = df_bearish_pts
    df['date'] = df_datetime_date
    
    #2.3: SEARCHING FOR FRACTAL EXECUTION CANDLES
    exe = []
    pct_chg = []
    
    
    for index, row in df.iterrows():
        if row['close'] > row['high fractal pts']:
            exe.append(1)
            pct_chg.append((row['close'] / row['high fractal pts'] - 1) * 100)
        elif row['close'] < row['low fractal pts']:
            exe.append(2)
            pct_chg.append((1 - (row['close'] / row['low fractal pts'])) * 100)
        else:
            exe.append(0)
            pct_chg.append(0)
    
    df['exe'] = exe
    df['pct chg'] = pct_chg
    
    #2.4: GENERATING EMA, ATR & EMA FILTER
    df['ema20'] = talib.EMA(df['close'],20)        
    df['atr5'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=5)
    
    ema_close = []
    
    for index, row in df.iterrows():
        if pd.isna(row['ema20']):
            ema_close.append(0)
        else:
            if row['exe'] == 1 and row['close'] > row['ema20']:
               ema_close.append(1)             
            elif row['exe'] == 2 and row['close'] < row['ema20']:
               ema_close.append(1)
            else:
               ema_close.append(0)
    
    df['ema_close'] = ema_close
    print(symbol+' done')
    return df


#PART 3: CONSOLIDATING & SAVING DATA

const = pd.read_csv('constituents_dj30.csv')
portfolio_df = []

for x in const.iloc[:,0]:
    symbol = x.lower()
    if len(portfolio_df) == 0:
        portfolio_df = process_data(symbol)
    else:
        portfolio_df = pd.concat([portfolio_df,process_data(symbol)], ignore_index=True)


import pickle

#Save Model
with open("portfolio_df_10Y_combined_dj30.pkl", "wb") as file: pickle.dump(portfolio_df, file)

#Load Model
portfolio_df = pickle.load(open( "portfolio_df_10Y.pkl", "rb" ))
