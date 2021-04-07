from binance.client import Client
from binance.websockets import BinanceSocketManager
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import btalib
import dateparser
import pytz
from datetime import datetime
import pickle
import csv
# Initialize Client

# ---- GETTING DATA -----
keysFile = open("keys.json")
keys = json.load(keysFile)
api_key = keys["binance"]["apiKey"]
api_secret = keys["binance"]["secret"]
client = Client(api_key, api_secret)
symbol = "ETHUSDT"
start = "1 Jan, 2018"
end = "1 Apr, 2021"
interval = Client.KLINE_INTERVAL_1HOUR
historicalData = client.get_historical_klines(symbol, interval, start, end)
header = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QouteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', '?']
with open("rawHistoricalData_{}_{}_{}-{}.csv".format(symbol, interval, start, end), 'w', newline = '' ) as fp:
    writer = csv.writer(fp)
    writer.writerow(i for i in header)
    writer.writerows(historicalData)

# CLEAN DATA
# delete unwanted data - just keep OpenTime, open, high, low, close, volume, 
for line in historicalData:
    del line[6:]
# Convert the epoch timestamp to regular date
pdHeader = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume']
df = pd.DataFrame(historicalData, columns=pdHeader)
df["OpenTime"] = df["OpenTime"] / 1000
df["Date"] = pd.to_datetime(df["OpenTime"], unit="s")
df["Close"] = pd.to_numeric(df["Close"])
print(df.head())
df.to_csv("cleanHistoricalData_{}_{}_{}-{}.csv".format(symbol, interval, start, end), index=False)