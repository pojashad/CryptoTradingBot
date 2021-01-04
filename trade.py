import datetime
import json
from binance.client import Client
import streamlit as st
import pandas as pd
import uuid
import os

class Trade(object):
    def __init__(self):
        self.commission = 0.001
        self.buyPrice = 0
        self.profitMargin = 1.01
        self.dollarWallet = 700
        self.cryptoQuantity = 0

    def buy(self, currentPrice):
        print("--Buying--")
        print(datetime.datetime.now())
        self.buyPrice = currentPrice + currentPrice * self.commission
        print(
            "Bought at:  ", currentPrice, "To beat: ", self.buyPrice * self.profitMargin
        )
        self.cryptoQuantity = (self.dollarWallet * 0.8) / (
            currentPrice + currentPrice * self.commission
        )
        self.cryptoQuantity = round(self.cryptoQuantity, 5)
        print("Bought ETH: ", self.cryptoQuantity)
        self.dollarWallet = self.dollarWallet - self.cryptoQuantity * currentPrice
        print("Wallet: ", self.dollarWallet)
        

        ## write to file
        buyInfo = {
            "UUID": str(uuid.uuid4()),
            "timestamp" : datetime.datetime.now().timestamp(),
            "dateTime" : str(datetime.datetime.now()),
            "symbol": "ETHBUSD",
            "buyPrice": self.buyPrice,
            "quantity": float(self.cryptoQuantity),
            "buySignal": "Below EMA200"
        }
        with open('activeOrders.json', 'w') as file:
            array = []
            array.append(buyInfo)
            json.dump(array, file)
       


        """
    activeOrders.json
        [{
            UUID: 3bb2ab84-4b54-11eb-ae93-0242ac130002
            dateTime: "2020-12-05 08:52:00"
            buyPrice: row['Close'] + (row['Close']*commission)
            crypto: (dollarWallet*0.8)/ \
                       (row['Close'] + (row['Close']*commission))
        }]
    """

    def sell(self, currentPrice):
        with open('activeOrders.json', 'r+') as json_file:
            activeOrders = json.load(json_file)
            print(activeOrders)
        for orders in activeOrders:
            cryptoQuantity = float(orders['quantity'])
            print(activeOrders)
            print("--Selling--")
            print(datetime.datetime.now())
            sellPrice = currentPrice
            print("Sold at: ", currentPrice)
            # Purchase based on dollarWallet balance = ränta på ränta effekten
            self.dollarWallet = self.dollarWallet + (
                cryptoQuantity * (currentPrice) - (cryptoQuantity * (currentPrice * self.commission))
            )
            print("Wallet: ", self.dollarWallet)
            
            
        
        
        
        """
    def evaluateSell(price):
        if(activeOrders > 0):
            for order in activeOrders:
                if(order.buyPrice*vinstMarginal < (price - (price*commission)) and currentPrice < previousPrice):
                    tradeSell(order)
"""