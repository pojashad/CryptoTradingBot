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
# ---- GETTING DATA -----
# Initialize Client
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

"""
historicalData = 
  [
  1514764800000, # Open time
  "733.01000000", # Open
  "734.52000000", # High
  "720.03000000", # Low
  "727.62000000", # Close
  "2105.90100000", # Volume
  1514768399999, # Close time
  "1528559.14512900", # Quote asset volume
  3114, # Number of trades
  "1275.23271000", # Taker buy base asset volume
  "925445.06828040",  # Taker buy quote asset volume
  "0"
]
  """
# open a file with filename including symbol, interval and start and end converted to milliseconds
header = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'QouteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', '?']
with open("historicalData_{}_{}_{}-{}.csv".format(symbol, interval, start, end), 'w', newline = '' ) as fp:
    writer = csv.writer(fp)
    writer.writerow(i for i in header)
    writer.writerows(historicalData)


