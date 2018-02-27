
# coding: utf-8

# ## Default

# In[1]:


import datetime
import pandas as pd
import numpy as np
import json
import oandapyV20
import credentials
import time
import os


# ## Helper Functions

# In[2]:


import pickle
def load_data(filename):
    try:
        with open(filename, 'rb') as f:
            x = pickle.load(f)
    except:
        print("Exception Occurred")
    return x
def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
def earliest_time(filename):
    '''
    returns the first time entry in streamer file
    '''
    with open(filename) as f:
        next(f)
        return pd.to_datetime(f.readline().split(',')[0])
def time_floor(time = '2018-01-31 03:20', floor_to = '30m'):
    '''
    takes date string and floors based on minutes or hours
    '''
    def floor_all(x, time_range, floor):
        return str(pd.cut([int(x)], bins = range(-1, time_range+1, floor), labels = range(0,time_range,floor))[0]).zfill(2)
    floor_int = int(floor_to[:-1])
    if floor_to[-1] == 'm':
        return time[:-2] + floor_all(time[-2:], 60, floor_int)
    if floor_to[-1] == 'h':
        return time[:-5] + floor_all(time[-5:-3], 24, floor_int) + ':00'
def line_convert(data, floor_to = None):
    '''
    Takes stream data-ticks and flattens them for file writing
    '''
    time = data['time'][:19]
    if floor_to:
        time = time_floor(time[:-3], floor_to)
    return ','.join([time, data['instrument'],
                    str(data['asks'][0]['liquidity']), data['asks'][0]['price'], 
                    str(data['bids'][0]['liquidity']), data['bids'][0]['price'], 
                        data['closeoutAsk'],
                        data['closeoutBid'],
                        data['status'],
                        data['type']])


# ## Access Functions

# In[3]:


import oandapyV20.endpoints.accounts as accounts
def access(access_token):
    return oandapyV20.API(access_token=access_token)


# ### Example
# ```python
# client = access(credentials.access_token)
# ```

# ## Historical Data

# In[4]:


import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.factories import InstrumentsCandlesFactory


# In[5]:


def historical_data(instrument, params,
                    client = access(credentials.access_token),
                    file_prefix = None):
    '''
    instrument = "GBP_USD"
    params = {
        "from": "2017-01-01T00:00:00Z",
        "granularity": "M30",
    }    
    '''
    def cnv(r, h):
        # get all candles from the response and write them as a record to the filehandle h
        for candle in r.get('candles'):
            ctime = candle.get('time')[:16]
            try:
                rec = "{time},{time_dec},{o},{h},{l},{c},{v}".format(
                    time=ctime.replace('T', ' ').replace('Z', ''),
                    time_dec=float(ctime.replace('T', ' ').replace('Z', '').replace(':', '.')[-5:])/24,
                    o=candle['mid']['o'],
                    h=candle['mid']['h'],
                    l=candle['mid']['l'],
                    c=candle['mid']['c'],
                    v=candle['volume'],
                )
            except Exception as e:
                print(e, r)
            else:
                h.write(rec+"\n")

    datafile = "tmp/{}_{}_{}.bulk".format(file_prefix,instrument, params['granularity'])
    with open(datafile, "w") as O:
        O.write('time,time_dec,Open,High,Low,Close,Volume\n')
        n = 0
        for r in InstrumentsCandlesFactory(instrument=instrument, params=params):
            rv = client.request(r)
            cnt = len(r.response.get('candles'))
            #print("REQUEST: {} {} {}, received: {}".format(r, r.__class__.__name__, r.params, cnt), end = '\r')
            n += cnt
            cnv(r.response, O)
        #print("Check the datafile: {} under /tmp!, it contains {} records".format(datafile, n))
    return r.params   


# ### Example
# 
# ```python
# instrument = "GBP_USD"
# params = {
#     "from": "2017-01-01T00:00:00Z",
#     "granularity": "M30",
# }
# historical_data(instrument, params=params, file_prefix = 'test1')
# ```

# ## Real-Time Functions

# In[6]:


import oandapyV20.endpoints.pricing as pricing
from oandapyV20.exceptions import StreamTerminated


# In[24]:


def current_price(instruments, accountID = credentials.accountID, 
                  client = access(credentials.access_token)):
    r = pricing.PricingStream(accountID = accountID, params = {"instruments": ",".join(instruments)})
    data_return = []
    try:
        for tick in client.request(r):
            if 'instrument' in tick:
                data_return.append(tick)
                instruments.remove(tick['instrument'])
            if len(instruments) < 1:
                r.terminate()
    except StreamTerminated as err:
        pass
    return data_return
def server_time():
    t = current_price(['GBP_USD'])[0]['time'][:19]
    return datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S')


# ### Example
# 
# ```python
# current_price(['GBP_USD', 'EUR_USD'])
# ```
# out:
# ```
# [{'asks': [{'liquidity': 10000000, 'price': '1.24302'}],
#   'bids': [{'liquidity': 10000000, 'price': '1.24285'}],
#   'closeoutAsk': '1.24317',
#   'closeoutBid': '1.24270',
#   'instrument': 'EUR_USD',
#   'status': 'tradeable',
#   'time': '2018-01-26T21:45:02.360446771Z',
#   'tradeable': True,
#   'type': 'PRICE'},
#  {'asks': [{'liquidity': 10000000, 'price': '1.41535'}],
#   'bids': [{'liquidity': 10000000, 'price': '1.41507'}],
#   'closeoutAsk': '1.41560',
#   'closeoutBid': '1.41482',
#   'instrument': 'GBP_USD',
#   'status': 'tradeable',
#   'time': '2018-01-26T21:45:02.296571250Z',
#   'tradeable': True,
#   'type': 'PRICE'}]
# ```

# In[8]:


def start_stream(filename, buffer_size = 60, total_time = np.inf,
                 client = access(credentials.access_token),
                 accountID = credentials.accountID,
                 instruments = ['GBP_USD'], 
                 floor_to = None):
    r = pricing.PricingStream(accountID = accountID, params = {"instruments": ",".join(instruments)})
    start_time = datetime.datetime.now()
    earliest_row = datetime.datetime.utcnow()+datetime.timedelta(seconds=200)
    #print (start_time)
    n = 1
    with open(filename, 'w') as f:
        f.write("time,instrument,asks_liquidity,asks_price,bids_liquidity,bids_price,closeoutAsk,closeoutBid,status,type\n")
    try:
        for tick in client.request(r):
            now = datetime.datetime.strptime(tick['time'][:19], "%Y-%m-%dT%H:%M:%S")
            #print (now)
            time_since = (now-earliest_row).total_seconds()
            #print (time_since)
            if time_since >= total_time:
                r.terminate()
            else:
                if tick['type'] == 'PRICE':
                    n += 1
                    if time_since <= buffer_size:
                        with open(filename, 'a') as f:
                            f.write("{}\n".format(line_convert(tick, floor_to)))
                    else:
                        with open(filename, 'r+') as f:
                            with open(filename, 'r+') as d:
                                f.write("time,instrument,asks_liquidity,asks_price,bids_liquidity,bids_price,closeoutAsk,closeoutBid,status,type\n")
                                line = d.readline()
                                line = d.readline()
                                for line in d:
                                    f.write(line)
                                f.write("{}\n".format(line_convert(tick)))
                    earliest_row = earliest_time(filename)
                    print ("{} price ticks. {} seconds total time active.".format(n, (datetime.datetime.now()-start_time).total_seconds()), end = '\r')
    except StreamTerminated as err:
        print ("Stream as reached total time.  Terminating.")


# ### Example: 30-minute time frame from streamed data with `pandas`
# 
# ```python
# pd.read_csv('tmp/stream14d_inf', parse_dates = ['time'], index_col = 'time')['asks_price'].resample('30min').ohlc()
# ```

# ## Trading

# In[9]:


import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    TakeProfitDetails,
    StopLossDetails)
import oandapyV20.endpoints.positions as positions


# In[10]:


def market_order(instrument, units, take_profit, stop_loss, accountID = credentials.accountID, 
                 client = access(credentials.access_token)):
    mktOrder = MarketOrderRequest(instrument = instrument,
                                 units = units,
                                 takeProfitOnFill=TakeProfitDetails(price=take_profit).data,
                                 stopLossOnFill=StopLossDetails(price=stop_loss).data
                                 ).data
    r = orders.OrderCreate(accountID=credentials.accountID, data = mktOrder)
    response = client.request(r)
    return mktOrder, r, response


# In[42]:


def close_position(instrument, data = None, accountID = credentials.accountID,
                   client = access(credentials.access_token)):
    '''
    data : {'shortUnits' : 'ALL'} or {'longUnits' : 'ALL'}
    '''
    if not data:
        return 'Please enter "data" parameters'
    r = positions.PositionClose(accountID=accountID,instrument=instrument,data=data)
    rv = client.request(r)
    return r, rv


# In[21]:


def check_positions(client = access(credentials.access_token),
                    accountID = credentials.accountID):
    r = positions.OpenPositions(accountID=credentials.accountID)
    rv = client.request(r)
    return r, rv


# ## Example
# 
# ### Opening Order
# 
# ```python
# price = float(current_price(['GBP_USD'])[0]['asks'][0]['price'])
# mktOrder, r, response = market_order('GBP_USD', 1000, price+0.004, price-0.002)
# ```
# `mktOrder`
# out:
# ```
# {'order': {'instrument': 'GBP_USD',
#   'positionFill': 'DEFAULT',
#   'stopLossOnFill': {'price': '1.41760', 'timeInForce': 'GTC'},
#   'takeProfitOnFill': {'price': '1.41270', 'timeInForce': 'GTC'},
#   'timeInForce': 'FOK',
#   'type': 'MARKET',
#   'units': '-1000'}}
# ```
# `dir(r)` methods
# out:
# ```
# ['ENDPOINT',
#  'EXPECTED_STATUS',
#  'HEADERS',
#  'METHOD',
#  '__abstractmethods__',
#  '__class__',
#  '__delattr__',
#  '__dict__',
#  '__dir__',
#  '__doc__',
#  '__eq__',
#  '__format__',
#  '__ge__',
#  '__getattribute__',
#  '__gt__',
#  '__hash__',
#  '__init__',
#  '__init_subclass__',
#  '__le__',
#  '__lt__',
#  '__module__',
#  '__ne__',
#  '__new__',
#  '__reduce__',
#  '__reduce_ex__',
#  '__repr__',
#  '__setattr__',
#  '__sizeof__',
#  '__str__',
#  '__subclasshook__',
#  '__weakref__',
#  '_abc_cache',
#  '_abc_negative_cache',
#  '_abc_negative_cache_version',
#  '_abc_registry',
#  '_endpoint',
#  '_expected_status',
#  '_response',
#  '_status_code',
#  'data',
#  'expected_status',
#  'method',
#  'response',
#  'status_code']
# ```
# `response`
# out:
# ```
# {'lastTransactionID': '45',
#  'orderCreateTransaction': {'accountID': '101-001-7534004-002',
#   'batchID': '42',
#   'id': '42',
#   'instrument': 'GBP_USD',
#   'positionFill': 'DEFAULT',
#   'reason': 'CLIENT_ORDER',
#   'requestID': '60405968221462696',
#   'stopLossOnFill': {'price': '1.41760', 'timeInForce': 'GTC'},
#   'takeProfitOnFill': {'price': '1.41270', 'timeInForce': 'GTC'},
#   'time': '2018-01-26T21:47:04.202600577Z',
#   'timeInForce': 'FOK',
#   'type': 'MARKET_ORDER',
#   'units': '-1000',
#   'userID': 7534004},
#  'orderFillTransaction': {'accountBalance': '10000.9207',
#   'accountID': '101-001-7534004-002',
#   'batchID': '42',
#   'commission': '0.0000',
#   'financing': '0.0000',
#   'fullPrice': {'asks': [{'liquidity': '10000000', 'price': '1.41510'}],
#    'bids': [{'liquidity': '10000000', 'price': '1.41486'}],
#    'closeoutAsk': '1.41535',
#    'closeoutBid': '1.41461',
#    'timestamp': '2018-01-26T21:47:04.168582950Z'},
#   'gainQuoteHomeConversionFactor': '1',
#   'guaranteedExecutionFee': '0.0000',
#   'halfSpreadCost': '0.1200',
#   'id': '43',
#   'instrument': 'GBP_USD',
#   'lossQuoteHomeConversionFactor': '1',
#   'orderID': '42',
#   'pl': '0.0000',
#   'price': '1.41486',
#   'reason': 'MARKET_ORDER',
#   'requestID': '60405968221462696',
#   'time': '2018-01-26T21:47:04.202600577Z',
#   'tradeOpened': {'guaranteedExecutionFee': '0.0000',
#    'halfSpreadCost': '0.1200',
#    'price': '1.41486',
#    'tradeID': '43',
#    'units': '-1000'},
#   'type': 'ORDER_FILL',
#   'units': '-1000',
#   'userID': 7534004},
#  'relatedTransactionIDs': ['42', '43', '44', '45']}
# ```
# 
# out if order did not place:
# ```
# {'lastTransactionID': '81',
#  'orderCancelTransaction': {'accountID': '101-001-7534004-002',
#   'batchID': '80',
#   'id': '81',
#   'orderID': '80',
#   'reason': 'TAKE_PROFIT_ON_FILL_LOSS',
#   'requestID': '24381349578710306',
#   'time': '2018-02-07T10:30:27.367090618Z',
#   'type': 'ORDER_CANCEL',
#   'userID': 7534004},
#  'orderCreateTransaction': {'accountID': '101-001-7534004-002',
#   'batchID': '80',
#   'id': '80',
#   'instrument': 'GBP_USD',
#   'positionFill': 'DEFAULT',
#   'reason': 'CLIENT_ORDER',
#   'requestID': '24381349578710306',
#   'stopLossOnFill': {'price': '1.39166', 'timeInForce': 'GTC'},
#   'takeProfitOnFill': {'price': '1.38566', 'timeInForce': 'GTC'},
#   'time': '2018-02-07T10:30:27.367090618Z',
#   'timeInForce': 'FOK',
#   'type': 'MARKET_ORDER',
#   'units': '1000',
#   'userID': 7534004},
#  'relatedTransactionIDs': ['80', '81']})
# ```
# 
# ### Closing Order
# 
# ```python
# r, rv = close_position('GBP_USD', data = {'shortUnits' : 'ALL'})
# ```
# 
# `rv`
# out:
# ```
# {'lastTransactionID': '50',
#  'relatedTransactionIDs': ['47', '48', '49', '50'],
#  'shortOrderCreateTransaction': {'accountID': '101-001-7534004-002',
#   'batchID': '47',
#   'id': '47',
#   'instrument': 'GBP_USD',
#   'positionFill': 'REDUCE_ONLY',
#   'reason': 'POSITION_CLOSEOUT',
#   'requestID': '24377172917404441',
#   'shortPositionCloseout': {'instrument': 'GBP_USD', 'units': 'ALL'},
#   'time': '2018-01-26T21:53:53.970213475Z',
#   'timeInForce': 'FOK',
#   'type': 'MARKET_ORDER',
#   'units': '1000',
#   'userID': 7534004},
#  'shortOrderFillTransaction': {'accountBalance': '10000.4008',
#   'accountID': '101-001-7534004-002',
#   'batchID': '47',
#   'commission': '0.0000',
#   'financing': '0.0001',
#   'fullPrice': {'asks': [{'liquidity': '10000000', 'price': '1.41538'}],
#    'bids': [{'liquidity': '10000000', 'price': '1.41480'}],
#    'closeoutAsk': '1.41563',
#    'closeoutBid': '1.41455',
#    'timestamp': '2018-01-26T21:53:53.912243825Z'},
#   'gainQuoteHomeConversionFactor': '1',
#   'guaranteedExecutionFee': '0.0000',
#   'halfSpreadCost': '0.2900',
#   'id': '48',
#   'instrument': 'GBP_USD',
#   'lossQuoteHomeConversionFactor': '1',
#   'orderID': '47',
#   'pl': '-0.5200',
#   'price': '1.41538',
#   'reason': 'MARKET_ORDER_POSITION_CLOSEOUT',
#   'requestID': '24377172917404441',
#   'time': '2018-01-26T21:53:53.970213475Z',
#   'tradesClosed': [{'clientTradeID': '194204054',
#     'financing': '0.0001',
#     'guaranteedExecutionFee': '0.0000',
#     'halfSpreadCost': '0.2900',
#     'price': '1.41538',
#     'realizedPL': '-0.5200',
#     'tradeID': '43',
#     'units': '1000'}],
#   'type': 'ORDER_FILL',
#   'units': '1000',
#   'userID': 7534004}}
# ```

# ### Checking Open Positions
# 
# ```python
# r, rv = check_positions()
# ```
# 
# `r, rv` 
# 
# for no positions
# out:
# ```
# (<oandapyV20.endpoints.positions.OpenPositions at 0x26036531be0>,
#  {'lastTransactionID': '50', 'positions': []})
# ```
# 
# for open long position
# out:
# ```
# (<oandapyV20.endpoints.positions.OpenPositions at 0x26036549080>,
#  {'lastTransactionID': '65',
#   'positions': [{'commission': '0.0000',
#     'financing': '0.0048',
#     'guaranteedExecutionFees': '0.0000',
#     'instrument': 'GBP_USD',
#     'long': {'averagePrice': '1.39675',
#      'financing': '0.0000',
#      'guaranteedExecutionFees': '0.0000',
#      'pl': '0.0000',
#      'resettablePL': '0.0000',
#      'tradeIDs': ['62'],
#      'units': '1000',
#      'unrealizedPL': '-0.1000'},
#     'marginUsed': '69.8370',
#     'pl': '0.3960',
#     'resettablePL': '0.3960',
#     'short': {'financing': '0.0048',
#      'guaranteedExecutionFees': '0.0000',
#      'pl': '0.3960',
#      'resettablePL': '0.3960',
#      'units': '0',
#      'unrealizedPL': '0.0000'},
#     'unrealizedPL': '-0.1000'}]})
# ```
# 
# for open short position
# out:
# ```
# (<oandapyV20.endpoints.positions.OpenPositions at 0x260365491d0>,
#  {'lastTransactionID': '75',
#   'positions': [{'commission': '0.0000',
#     'financing': '0.0044',
#     'guaranteedExecutionFees': '0.0000',
#     'instrument': 'GBP_USD',
#     'long': {'financing': '-0.0004',
#      'guaranteedExecutionFees': '0.0000',
#      'pl': '-0.8700',
#      'resettablePL': '-0.8700',
#      'units': '0',
#      'unrealizedPL': '0.0000'},
#     'marginUsed': '69.7975',
#     'pl': '-0.4740',
#     'resettablePL': '-0.4740',
#     'short': {'averagePrice': '1.39574',
#      'financing': '0.0048',
#      'guaranteedExecutionFees': '0.0000',
#      'pl': '0.3960',
#      'resettablePL': '0.3960',
#      'tradeIDs': ['72'],
#      'units': '-1000',
#      'unrealizedPL': '-0.3200'},
#     'unrealizedPL': '-0.3200'}]})
# ```
