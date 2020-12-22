import ccxt
import json
keysFile = open ('keys.json')
keys = json.load(keysFile)
print(keys)
# from variable id
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': keys['apiKey'],
    'secret': keys['secret'],
    'timeout': 30000,
    'enableRateLimit': True,
})

#print(exchange.fetch_balance())