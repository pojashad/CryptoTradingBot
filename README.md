# CryptoTradingBot
Buy low, sell high? That sounds easy right? 
I am an easy subject of my feelings and when shit starts to hit the fan. I make dumb decisions. How about getting rid of my feelings and let a bot do the work for me? 

The main purpose of this project is to learn/get better at:
- Python
- Crypto
- Trading Crypto 
    - Moving Average & Exponential Moving Average
- Pandas & Numpy

And if i could make some passive income on this (REALLY LOW ODDS OF THAT HAPPENING) i could take my brother out for pizza. 

Why crypto? Because it trades 24/7 and i am not restricted to specific times and days. 
## Pre-requisites
### - [x] Find Crypto Exchange that has an API.

Binance has established itself as a market leader when it comes to cryptocurrency trading. It currently ranks number one for Bitcoin volume according to coinmarketcap.com and ranks well for many other currencies.

Commissions are very competitive and you may be hard-pressed to find another exchange with lower fees.

Lastly, Binance has a good track record of security. There have only been a few instances of Binance getting hacked which is something that all exchanges are susceptible to due to the nature of the business.

### - [x] Python libraries for the Binance API
CCXT A JavaScript / Python / PHP library for cryptocurrency trading and e-commerce with support for many bitcoin/ether/altcoin exchange markets and merchant APIs.

- [ ] Choose which Crypto to "trade"


# Findings
From Investopedia:
## Simple Moving Average & Exponential Moving Average

### SMA
The simplest form of a moving average, known as a simple moving average (SMA), is calculated by taking the arithmetic mean of a given set of values. In other words, a set of numbers–or prices in the case of financial instruments–are added together and then divided by the number of prices in the set. The formula for calculating the simple moving average of a security is as follows:

![SMA](/images/sma.png)

Lets say we have a SMA of 3 days. 

The price for 3 days have been:
- Day 1 = 25
- Day 2 = 26
- Day 3 = 27

![SMA](./images/sma_3.png)
| days = n  | Closing price | SMA_3|   |   |
|-------------|---------------|-----------------------|---|---|
| 1           | 25            | x                         |   |   |
| 2           | 26            | x                         |   |   |
| 3           | 27            | x                         |   |   |
| 4           | 25            | 26                       |   |   |

https://www.datacamp.com/community/tutorials/moving-averages-in-pandas

### EMA 

# Trading Strategy
First we need to do EMA and SMA on our historical data. The one i went with is EMA for 500 samples. We also need to know our buying price - which is the price we are buying at plus the commission of the trade. Binance has a commission of 0.01 % . That means if we buy ETH at 600 dollar the commission is 6 dollar. So the total buying price will be 606 dollars. So when we sell we need to beat the buying price and the commission of the sale.

![SMA](./images/buyAndSell1.png)
## When to buy?
 When the price is below the exponential moving average we are potentially in a buy position and when the price is increasing we are going to enter a buy position.
# !!! Disclaimer !!!

This software is for educational purposes only. Do not risk money which you are not ready to lose. USE THE SOFTWARE AT YOUR OWN RISK. I WILL NOT ASSUME RESPONSIBILITY FOR YOUR LOSSES.

NEVER RUN THE SOFTWARE IN LIVE MODE IF YOU'RE NOT 100% SURE OF WHAT YOU ARE DOING. Once again, I will NOT take ANY responsibilities regarding the results of your trades.

As is, the software is obviously not viable, but with some research and a bit of coding, it could become your best ally in automating strategies.
