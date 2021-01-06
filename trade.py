import datetime
import json
import uuid
import pprint

class Trade(object):
    def __init__(self):
        self.commission = 0.001
        self.buyPrice = 0
        self.profitMargin = 1.01
        self.dollarWallet = 700
        self.cryptoQuantity = 0
        self.array = []
        self.pp = pprint.PrettyPrinter(indent=4)

    def buy(self, currentPrice):
        print("--Buying--")
        self.buyPrice = currentPrice + currentPrice * self.commission
        self.cryptoQuantity = (self.dollarWallet * 0.8) / (
            currentPrice + currentPrice * self.commission
        )
        self.cryptoQuantity = round(self.cryptoQuantity, 5)
        buyInfo = {
            "UUID": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().timestamp(),
            "dateTime": str(datetime.datetime.now()),
            "symbol": "ETHBUSD",
            "buyPrice": self.buyPrice,
            "sellQuantityAt": self.buyPrice*self.profitMargin,
            "quantity": float(self.cryptoQuantity),
            "buySignal": "Below EMA200",
        }
        # write to file
        self.pp.pprint(buyInfo)
        with open("activeOrders.json", "w") as file:
            self.array.append(buyInfo)
            json.dump(self.array, file)
        self.dollarWallet = self.dollarWallet - self.cryptoQuantity * currentPrice
        print("Wallet: ", self.dollarWallet)

        

    def sell(self, currentPrice):
        with open("activeOrders.json", "r+") as json_file:
            activeOrders = json.load(json_file)
        
        indexToPop = []
        for index, orders in enumerate(activeOrders):
            cryptoQuantity = float(orders["quantity"])
            print("--Selling--")
            print(datetime.datetime.now())
            # sellPrice = currentPrice
            print("Sold at: ", currentPrice)
            # Purchase based on dollarWallet balance = ränta på ränta effekten
            self.dollarWallet = self.dollarWallet + (
                cryptoQuantity * (currentPrice)
                - (cryptoQuantity * (currentPrice * self.commission))
            )
            print("Wallet: ", self.dollarWallet)
            indexToPop.append(index)

        for index in indexToPop:    
            activeOrders.pop(index)
            
        with open("activeOrders.json", "w") as file:
            json.dump(activeOrders, file)

        """
    def evaluateSell(price):
        if(activeOrders > 0):
            for order in activeOrders:
                if(order.buyPrice*vinstMarginal < (price - (price*commission)) and currentPrice < previousPrice):
                    tradeSell(order)
"""
