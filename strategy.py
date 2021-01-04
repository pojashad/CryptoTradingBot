import pandas as pd
import numpy as np
import datetime
from trade import Trade


class Strategy(object):
    def __init__(self):
        self.trade = Trade()
        self.previousPrice = 0
        self.bought = False
        self.commission = 0.001
        self.buyPrice = 0
        self.profitMargin = 1.01
        self.priceArray = []
        self.EMA200 = 0
        self.dollarWallet = 700
        self.ETHs = 0

    def tick(self, currentPrice):
        # print(currentPrice)
        # Technical Analysis
        self.TA(currentPrice)

        # Evaluate if the current price and the technical Analysis should trigger a buy
        self.evaluateBuy(currentPrice)
        self.evaluateSell(currentPrice)
        self.previousPrice = currentPrice

    # Technical Analysis
    def TA(self, currentPrice):
        self.priceArray.append(currentPrice)
        ## EMA200
        if len(self.priceArray) > 200:
            self.priceArray.pop(0)
            df_test = pd.DataFrame(self.priceArray, columns=["CurrentPrice"])
            df_test["EMA200"] = df_test["CurrentPrice"].ewm(span=200).mean()
            self.EMA200 = df_test.at[len(df_test) - 1, "EMA200"]

    def evaluateBuy(self, currentPrice):
        # When the price is below the moving average we are potentially in a buy position.
        # When the price is increasing we are going to enter a buy poisition.
        if (
            currentPrice < self.EMA200
            and currentPrice > self.previousPrice
            and self.ETHs == 0
        ):
            self.trade.buy(currentPrice)
            self.buyPrice = currentPrice + currentPrice * self.commission
            self.ETHs = (self.dollarWallet * 0.8) / (
                currentPrice + currentPrice * self.commission
            )

    def evaluateSell(self, currentPrice):
        # When the price rises over the MA we are in a potential sell position
        # When the price startes to decrease( latestClosingPrice < previousClosingPrice) we are going to enter a sell position.
        # Sell when the closing price is higher than the buyPrice(buying price + comission) + profit margin. Securing that the buying & Sell commission are returned and a profit margin
        margin = self.buyPrice * self.profitMargin
        triggerPrice = currentPrice - (currentPrice * self.commission)
        if (
            margin < triggerPrice
            and currentPrice < self.previousPrice
            and self.ETHs > 0
        ):
            self.trade.sell(currentPrice)
            self.ETHs = 0
