import pandas as pd
import uuid
import datetime
import pprint
import json
from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceOrderException
class Strategy(object):
    def __init__(self):
        self.pp = pprint.PrettyPrinter(depth=4)
        self.previousPrice = 0
        self.bought = False
        self.commission = 0.001
        self.buyPrice = 0
        self.profitMargin = 1.01
        self.priceArray = []
        self.EMA200 = 0
        self.dollarWallet = 0
        self.cryptoQuantity = 0
        self.trades = []
        self.keysFile = open("keys.json")
        keys = json.load(self.keysFile)
        api_key = keys["binance"]["apiKey"]
        api_secret = keys["binance"]["secret"]
        symbol = "ETHBUSD"
        self.client = Client(api_key, api_secret)

    def tick(self, currentPrice):
        # Technical Analysis
        self.TA(currentPrice)

        # Evaluate if the current price and the technical Analysis should trigger a buy
        self.evaluateBuy(currentPrice)
        # Evaluate if the current price and the profit margin should trigger a sell
        self.evaluateSell(currentPrice)

        self.previousPrice = currentPrice

    def getBalance(self):
            balance = self.client.get_asset_balance(asset='BUSD')
            return float(balance['free'])

    # Technical Analysis
    def TA(self, currentPrice):
        self.priceArray.append(currentPrice)
        # EMA200
        if len(self.priceArray) > 200:
            self.priceArray.pop(0)
            df_test = pd.DataFrame(self.priceArray, columns=["CurrentPrice"])
            df_test["EMA200"] = df_test["CurrentPrice"].ewm(span=200).mean()
            self.EMA200 = df_test.at[len(df_test) - 1, "EMA200"]

    def evaluateBuy(self, currentPrice):
        # When the price is below the moving average we are potentially in a buy position.
        # When the price is increasing we are going to enter a buy poisition.
        if (currentPrice < self.EMA200 and currentPrice > self.previousPrice and self.cryptoQuantity == 0):
            #self.trade.buy(currentPrice)
            balance = self.client.get_asset_balance(asset='BUSD')
            self.dollarWallet = float(balance['free'])
            self.cryptoQuantity = (self.dollarWallet * 0.8) / (currentPrice + (currentPrice * self.commission))
            self.cryptoQuantity = round(self.cryptoQuantity, 4)
            self.buyPrice = currentPrice + (currentPrice * self.commission)
            print('-- BUYING --')
            print(datetime.datetime.now())
            # FIRE ORDER
            try:
                buy = self.client.order_market_buy(symbol='ETHBUSD', quantity=self.cryptoQuantity)
            except BinanceAPIException as e:
                # error handling goes here
                print(e)
            except BinanceOrderException as e:
                # error handling goes here
                print(e)
            print(buy)
            print("Bought at current price:" , currentPrice, "bought total of", self.cryptoQuantity , "for with commission:" , self.cryptoQuantity*currentPrice,  "price to beat with profit margin:", self.buyPrice*self.profitMargin)
            self.dollarWallet = self.getBalance
            print("Wallet:", self.dollarWallet)

    def evaluateSell(self, currentPrice):
        # When the price rises over the MA we are in a potential sell position
        # When the price startes to decrease( latestClosingPrice < previousClosingPrice) we are going to enter a sell position.
        # Sell when the closing price is higher than the buyPrice(buying price + comission) + profit margin. Securing that the buying & Sell commission are returned and a profit margin
        triggerPrice = currentPrice - (currentPrice * self.commission)
        if (self.buyPrice*self.profitMargin < triggerPrice and currentPrice < self.previousPrice and self.cryptoQuantity > 0):
            print('-- Selling --')
            print(datetime.datetime.now())
            Profit = self.cryptoQuantity * (currentPrice -  (currentPrice * self.commission))
            # FIRE ORDER
            try:
                sell = self.client.order_market_sell(symbol='ETHBUSD', quantity=self.cryptoQuantity)
            except BinanceAPIException as e:
                # error handling goes here
                print(e)
            except BinanceOrderException as e:
                # error handling goes here
                print(e)
            print("Sold at current price:" , currentPrice , "Sold total of:", self.cryptoQuantity , "for:", self.cryptoQuantity*currentPrice,  "Profit minus the commission", Profit)
            balance = self.client.get_asset_balance(asset='BUSD')
            self.dollarWallet = float(balance['free'])
            print('Wallet: ', self.dollarWallet)
            self.cryptoQuantity = 0

        elif (currentPrice < self.buyPrice*0.95 and self.cryptoQuantity > 0): # Stop loss
            print('-- Selling on Stop Loss')
            print(datetime.datetime.now())
            Loss = self.cryptoQuantity * (currentPrice - (currentPrice * self.commission))
             # FIRE ORDER
            try:
                sell = self.client.order_market_sell(symbol='ETHBUSD', quantity=self.cryptoQuantity)
            except BinanceAPIException as e:
                # error handling goes here
                print(e)
            except BinanceOrderException as e:
                # error handling goes here
                print(e)
            balance = self.client.get_asset_balance(asset='BUSD')
            self.dollarWallet = float(balance['free'])
            print('Wallet: ', self.dollarWallet)
            self.cryptoQuantity = 0

        