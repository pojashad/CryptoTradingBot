from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import btalib
import datetime
import pickle

# ---- GETTING DATA -----
# Initialize Client
keysFile = open('keys.json')
keys = json.load(keysFile)
api_key = keys['binance']['apiKey']
api_secret = keys['binance']['secret']
symbol = 'ETHUSDT'

client = Client(api_key, api_secret)


# ---- GETTING HISTORICAL DATA -----
# Get the historical data from the earliest vaild timestamp
###timestamp = client._get_earliest_valid_timestamp(symbol, '1d')
"""
days = 30
timestamp = datetime.datetime.now().timestamp()-(days*86400)

samplingInterval = '1m'
historicalData = client.get_historical_klines(
    symbol, samplingInterval, str(timestamp), limit=1000)
# delete unwanted data - just keep date, open, high, low, close
for line in historicalData:
    del line[5:]


with open('historicalData.csv', 'wb') as fp:
    pickle.dump(historicalData, fp)
"""

# ---- CREATE DATAFRAME -----
# Create a pandas dataframe from the array
with open ('historicalData.csv', 'rb') as fp:
    historicalData = pickle.load(fp)
headers = ['Timestamp', 'Open', 'High', 'Low', 'Close']
df = pd.DataFrame(historicalData, columns=headers)

# Convert the epoch timestamp to regular date
df['Timestamp'] = df['Timestamp']/1000
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
df['Close'] = pd.to_numeric(df['Close'])
# ---- Adding Technical Indicators -----

# SMA Pandas & bta-lib
df['Closing_SMA_5'] = btalib.sma(df['Close'], period=5).df
df['Closing_SMA_50'] = btalib.sma(df['Close'], period=50).df
df['Closing_SMA_200'] = df['Close'].rolling(window=200).mean()

# EMA
df['Closing EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['Closing EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['Closing EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['Closing EMA_500'] = df['Close'].ewm(span=500, adjust=False).mean()

print(df.tail(30))
# TRADING STRATEGY: testing on historicalData
tradePlaced = False
typeOfTrade = False
dollarWallet = 10000
simpelWallet = 10000
ETHs = 0
Bought = False
PriceAndcommission = 0
commission = 0.01
vinstMarginal = 1.05
buyPrice = 0
print("--Start--")
print("Wallet: " , dollarWallet)
## Ränta på Ränta
df.loc[0, 'Wallet'] = dollarWallet
df['Wallet'] = pd.to_numeric(df['Wallet'])
## enkel
df.loc[0, 'simpelWallet'] = simpelWallet
df['simpelWallet'] = pd.to_numeric(df['simpelWallet'])
for index, row in df.iterrows():
    previousPrice = df.iloc[index-1]['Close']
    # When the price is below the moving average we are potentially in a buy position.
    # When the price is increasing we are going to enter a buy poisition.
    if(dollarWallet > 0):
        if(row['Close'] < row['Closing EMA_500'] and row['Close'] > previousPrice and ETHs == 0):
            print("--Buying--")
            buyPrice = row['Close'] + (row['Close']*commission)
            print("Bought at: " , row['Close'])
            df.loc[index, 'BuyOrder'] = buyPrice
            df.loc[index, 'TransactionPoint'] = row['Close']
            ## Purchase based on dollarWallet balance = ränta på ränta effekten
            ETHs = (dollarWallet*0.8)/(row['Close'] + (row['Close']*commission))
            print("Bought ETH: ", ETHs)
            dollarWallet =  dollarWallet - ETHs*row['Close']
            print("Wallet: ", dollarWallet)
            ## Simpel Wallet
            simpelWallet = simpelWallet - buyPrice
        # When the price rises over the MA we are in a potential sell position
        # When the price startes to decrease( latestClosingPrice < previousClosingPrice) we are going to enter a sell position.
        # Sell when the closing price is higher than the buyPrice(buying price + comission) + profit margin. Securing that the buying & Sell commission are returned and a profit margin 
        elif(buyPrice*vinstMarginal < (row['Close'] - (row['Close']*commission)) and row['Close'] < previousPrice and ETHs > 0):
            print("--Selling--")
            sellPrice = row['Close'] - (row['Close']*commission)
            df.loc[index, 'SellOrder'] = sellPrice
            df.loc[index, 'TransactionPoint'] = row['Close']
            print("Sold at: ", row['Close'])
            ## Purchase based on dollarWallet balance = ränta på ränta effekten
            dollarWallet =  dollarWallet + (ETHs*(row['Close'])-(ETHs*(row['Close'])*commission))
            df.loc[index, 'Wallet'] = dollarWallet
            print("Wallet: ", dollarWallet)
            ETHs = 0
            ## Simpel Wallet
            simpelWallet = simpelWallet + sellPrice
            df.loc[index, 'simpelWallet'] = simpelWallet
        if(dollarWallet < 0):
            print('Game Over')
    gotHistoricaldata = True

print(df.tail(20))
print(dollarWallet)
df.to_csv("trading.csv")

# ---- Visualize -----
plt.figure(figsize=[15, 10])
plt.grid(True)
# Closing Price
df['Close'].plot()
##df['Wallet'].plot(x=df.index, y=df.Wallet, marker="^")
##df['simpelWallet'].plot(x=df.index, y=df.simpelWallet, marker="^")
"""
# SMA of 5 days on closing price
df['Closing_SMA_5'].plot(label='Closing SMA 5')
# SMA of 50 days on closing price
df['Closing_SMA_50'].plot(label='Closing SMA 50')
# SMA of 200 days on closing price
df['Closing_SMA_200'].plot(label='Closing SMA 200')
"""
# EMA of 10 days on closing price
#df['Closing EMA_10'].plot(label='Closing EMA_10')
# EMA of 20 days on closing price
#df['Closing EMA_20'].plot(label='Closing EMA_20')
# EMA of 50 days on closing price
# df['Closing EMA_50'].plot(label='Closing EMA_50')
# EMA of 200 days on closing price
df['Closing EMA_500'].plot(label='Closing EMA_500')
df['BuyOrder'].plot(x=df.index, y=df.Close,
                    c="blue", marker="^",  label="Bought")
df['SellOrder'].plot(x=df.index, y=df.Close,
                     c="red", marker="^",  label="Sold")
df['TransactionPoint'].plot(x=df.index, y=df.Close,
                            c="black", marker="^",  label="Transaction")
##df['TransactionPoint'].plot(x=df.index, color='k', linestyle='--')
##print(df['TransactionPoint'])
##plt.vlines(x=df['TransactionPoint'],ymin = 600, ymax = 1200, colors='purple', ls='--', lw=2, label='vline_multiple')
plt.legend(loc=2)
plt.show()

# ---- GETTING LIVE DATA -----