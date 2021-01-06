from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pickle
import pandas as pd
from strategy import Strategy
import json

strategy = Strategy()

keysFile = open("keys.json")
keys = json.load(keysFile)
api_key = keys["binanceTest"]["apiKey"]
api_secret = keys["binanceTest"]["secret"]
symbol = "ETHBUSD"
client = Client(api_key, api_secret)
client.API_URL = "https://testnet.binance.vision/api"


def process_message(msg):
    if msg["e"] == "error":
        print("Error: ", msg)
        bm.close()
        bm.start()
    else:
        currentPrice = float(msg["c"])
        Timestamp = msg["E"]
        #print(currentPrice)
        strategy.tick(currentPrice)


bm = BinanceSocketManager(client)
# start any sockets here, i.e a trade socket
conn_key = bm.start_symbol_ticker_socket(symbol, process_message)
# then start the socket manager
bm.start()
"""
trt
    buy = client.create_test_order(symbol='ETHUSDT', side='BUY', type='MARKET', quantity=100)
except BinanceAPIException as e:
    # error handling goes here
    print(e)
except BinanceOrderException as e:
    # error handling goes here
    print(e)
print(buy)
"""

# BACKTESTING
with open("historicalData.csv", "rb") as fp:
    historicalData = pickle.load(fp)
headers = ["Timestamp", "Open", "High", "Low", "Close"]
df = pd.DataFrame(historicalData, columns=headers)
for index, row in df.iterrows():
    strategy.tick(float(row["Close"]))
    # print(row['Close'])
print("done")
