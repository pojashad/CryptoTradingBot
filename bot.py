import random
import ccxt
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
keysFile = open('keys.json')
keys = json.load(keysFile)
symbol = 'ETH/BUSD'  # Crypto symbol
exchange_id = 'binance'
t_frame = '1m'  # For historical data 1-day timeframe

# Connect to exchange
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
        'apiKey': keys[exchange_id]['apiKey'],
        'secret': keys[exchange_id]['secret'],
        'timeout': 30000,
        'enableRateLimit': True,
})

# Create historical dataframe and fill with historical data
historicalData = exchange.fetch_ohlcv(symbol, t_frame) ## CSV LIST
header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
historyDf = pd.DataFrame(historicalData, columns=header).set_index('Timestamp')
historyDf['Symbol'] = symbol
historyDf.index = historyDf.index/1000
historyDf['Date'] = pd.to_datetime(historyDf.index, unit='s')
OHLCVDataFrame = historyDf

# Fill dataframe with latest ticker data
loop = True
count = 0
i=0
x=list()
y=list()
while loop == True:
    print(exchange.fetch_order_book (symbol))
    """
    #Get the latest ticker data for the crypto symbol
    latestTickerData = exchange.fetch_ticker(symbol)
    #print(latestTickerData)

    # exchange.fetch_ticker(symbol) contains a lot of data. Get the relevant data
    latestTimestampEpoch = latestTickerData['timestamp']
    latestOpen = latestTickerData['open']
    latestHigh = latestTickerData['high']
    latestLow = latestTickerData['low']
    latestClose = latestTickerData['close']
    latestVolume = float(latestTickerData['info']['volume']) 
    latestDate = latestTimestampEpoch

    # Fill the relevant data to dataframe 
    latestOHLCV = [[latestTimestampEpoch, latestOpen, latestHigh,
                   latestLow, latestClose, latestVolume]] ## CSV LIST
    tempDataFrame = pd.DataFrame(latestOHLCV,  columns=header).set_index('Timestamp')
    tempDataFrame['Symbol'] = symbol
    tempDataFrame.index = tempDataFrame.index/1000
    tempDataFrame['Date'] = pd.to_datetime(tempDataFrame.index, unit='s')
    OHLCVDataFrame = pd.concat([OHLCVDataFrame, tempDataFrame])
    print(OHLCVDataFrame)
    """
    
    time.sleep(1)
    loop = False




