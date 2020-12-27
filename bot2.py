from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import btalib
import datetime

# ---- GETTING DATA -----
# Initialize Client
keysFile = open('keys.json')
keys = json.load(keysFile)
api_key = keys['binance']['apiKey']
api_secret = keys['binance']['secret']
symbol = 'ETHUSDT'

client = Client(api_key, api_secret)

# ---- GETTING DATA -----
# Get the historical data from the earliest vaild timestamp
##timestamp = client._get_earliest_valid_timestamp(symbol, '1d')
days = 365
timestamp = datetime.datetime.now().timestamp()-(days*86400)
samplingInterval = '1d'
historicalData = client.get_historical_klines(
    symbol, samplingInterval, str(timestamp), limit=1000)
# delete unwanted data - just keep date, open, high, low, close
for line in historicalData:
    del line[5:]

# ---- CREATE DATAFRAME -----
# Create a pandas dataframe from the array
headers = ['Timestamp', 'Open', 'High', 'Low', 'Close']
df = pd.DataFrame(historicalData, columns=headers)

# Convert the epoch timestamp to regular date
df['Timestamp'] = df['Timestamp']/1000
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')

# ---- Adding Technical Indicators -----

# SMA Pandas & bta-lib
df['Closing_SMA_5'] = btalib.sma(df['Close'], period=5).df
df['Closing_SMA_50'] = btalib.sma(df['Close'], period=50).df
df['Closing_SMA_200'] = df['Close'].rolling(window=200).mean()

# EMA
df['Closing EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['Closing EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['Closing EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

print(df.tail(30))


# TRADING STRATEGY
tradePlaced = False
typeOfTrade = ""
wallet = 0
for index, row in df.iterrows():
    previousPrice = float(df.iloc[index-1]['Close'])
    if tradePlaced == False:
        # When the price rises over the MA we are in a potential sell position
        # When the price startes to decrease( latestClosingPrice < previousClosingPrice) we are going to enter a sell position.
        if(float(row['Close']) > row['Closing EMA_10'] and float(row['Close']) < previousPrice):
            ##print('SELL ORDER')
            tradePlaced = True
            typeOfTrade = "short"
            df.loc[index, 'Order'] = 'SELL ORDER'
        # When the price is below the moving average we are potentially in a buy position.
        # When the price is increasing we are going to enter a buy poisition.
        elif(float(row['Close']) < row['Closing EMA_10'] and float(row['Close']) > previousPrice):
           ## print('BUY ORDER')
            tradePlaced = True
            typeOfTrade = "long"
            df.loc[index, 'Order'] = 'BUY ORDER'
    elif (typeOfTrade == "short"):
        if(float(row['Close']) < previousPrice):
             ## print("EXIT TRADE")
            tradePlaced = False
            typeOfTrade = ""
            df.loc[index, 'Transaction'] = 'SOLD'
    elif (typeOfTrade == "long"):
        if(float(row['Close']) > previousPrice):
            ##print("EXIT TRADE")
            tradePlaced = False
            typeOfTrade = ""
            df.loc[index, 'Transaction'] = 'BOUGHT'
           
print(df.tail(200))
print(wallet)
df.to_csv("trading.csv")
# With bta-lib

# ---- Visualize -----

plt.figure(figsize=[15,10])
plt.grid(True)
# Closing Price
df['Close']=pd.to_numeric(df['Close'])
df['Close'].plot()
# SMA of 5 days on closing price
df['Closing_SMA_5']=pd.to_numeric(df['Closing_SMA_5'])
df['Closing_SMA_5'].plot(label='Closing SMA 5')
# SMA of 50 days on closing price
df['Closing_SMA_50']=pd.to_numeric(df['Closing_SMA_50'])
df['Closing_SMA_50'].plot(label='Closing SMA 50')
# SMA of 200 days on closing price
df['Closing_SMA_200']=pd.to_numeric(df['Closing_SMA_200'])
df['Closing_SMA_200'].plot(label='Closing SMA 200')
# EMA of 10 days on closing price
df['Closing EMA_10']=pd.to_numeric(df['Closing EMA_10'])
df['Closing EMA_10'].plot(label='Closing EMA_10')
# EMA of 20 days on closing price
df['Closing EMA_20']=pd.to_numeric(df['Closing EMA_20'])
df['Closing EMA_20'].plot(label='Closing EMA_20')
# EMA of 50 days on closing price
df['Closing EMA_50']=pd.to_numeric(df['Closing EMA_50'])
df['Closing EMA_50'].plot(label='Closing EMA_50')

# Crossover
plt.legend(loc=2)
plt.show()


# When the price is below the moving average we are potentially in a buy position.
# When the price is increasing we are going to enter a buy poisition.
# When the price rises over the MA we are in a potential sell position
# When the price startes to decrease we are going to enter a sell position.

# GET LIVE DATA
"""
lastPrices = []
prices = []
currentMovingAverage = 0
lengthOfMA = 0
historicalData = False
tradePlaced = False
typeOfTrade = False
eth_price = {'error': False}


def eth_liveData(msg):
    ''' define how to process incoming WebSocket messages '''
    if msg['e'] != 'error':
        lastPairPrice = eth_price['last'] = msg['c']
        lastPrices.append(float(eth_price['last']))
                
        print ("{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()) + " Symbol: %s Price: %s Moving Average: %s" % (symbol, eth_price['last'], currentMovingAverage))

    else:
        eth_price['error'] = True

print(datetime.datetime.now().timestamp()-(30*86400) )
bsm = BinanceSocketManager(client)
conn_key = bsm.start_symbol_ticker_socket(symbol, eth_liveData)
bsm.start()
"""
