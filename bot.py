import ccxt

# from variable id
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': 'dOPgR5rwDTimeokpratr8osgvnFUF00CUrfcKmK6F2grpSNugL72nfXcXzQoXaTI',
    'secret': 'eJx96memxd1IMCIQ4qWwqXPtXenX3j2STAjhqWx09giwsxPCLjTOQrCAWxTnUkCd',
    'timeout': 30000,
    'enableRateLimit': True,
})

print(exchange.fetch_balance())