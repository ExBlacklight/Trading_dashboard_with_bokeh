from kiteconnect import KiteConnect,KiteTicker
import reqTokenTOTP as reqToken
import numpy as np
from datetime import datetime,time,timedelta
import os
import pickle
import pandas as pd
from time import sleep
import random
from collections import deque
from copy import deepcopy
from time import sleep
import numbers
import threading

class KT:
    def __init__(self):
        self.nifty50 = {'instrument_token':256265,'tradingsymbol':'NIFTY 50','name':'NIFTY 50','option_name':'NIFTY'}
        self.bank_nifty = {'instrument_token':260105,'tradingsymbol':'NIFTY BANK','name':'NIFTY BANK','option_name':'BANKNIFTY'}
        self.five_minute = {'name':'5minute','value':300}
        self.minute = {'name':'minute','value':60}
        self.channels = [self.bank_nifty]
        self.flag = None
        self.tokens = {}
        self.monitor = self.bank_nifty# Enter the trading symbol which has to be monitored
        self.interval = self.five_minute# Enter the interval HERE!!!
        self.channel_data = {}
        self.next_update = self.func()
        self.update_number=0
        self.queue = []
        self.remTime = None
        self.today = datetime.now().date()
        self.KiteStart()
        self.tickerConnection()
        self.startTicker()
        self.getTokens()
        self.queueThreadRunning=True
        self.threadRemTimeRunning=True
        self.queueThread = threading.Thread(target=self.queueThreadFunction,daemon=True)
        self.remTimeThread = threading.Thread(target=self.remainingTimeFunction,daemon=True)
        self.queueThread.start()
        self.remTimeThread.start()
        self.ce_order_queue = None
        self.pe_order_queue = None
        self.order_queue = []
        self.orders = []
        self.current_orders = []
    
    def KiteStart(self):
        self.api_key = '9l40nyckacd0aqw4'
        api_secret = 'y568sox3hfqhchqzp0v0wsu0aqpha345'
        self.kite = KiteConnect(api_key=self.api_key)
        requestToken = reqToken.get()
        self.data = self.kite.generate_session(requestToken,api_secret=api_secret)
        self.kite.set_access_token(self.data["access_token"])
        
    
    
    def tickerConnection(self):
        self.ticker = KiteTicker(self.api_key, self.data["access_token"],debug=True)
        self.ticker.on_ticks = self.on_ticks
        self.ticker.on_connect = self.on_connect
        self.ticker.on_reconnect = self.on_reconnect
        self.ticker.on_error = self.on_error
        self.ticker.on_close = self.on_close
        self.ticker.on_order_update = self.on_order_update
    
    def tickerReconnection(self):
        self.ticker = KiteTicker(self.api_key, self.data["access_token"],debug=True)
        self.ticker.on_ticks = self.on_ticks
        self.ticker.on_connect = self.on_connect
        self.ticker.on_reconnect = self.on_reconnect
        self.startTicker()
    
    
    def on_order_update(self,ws, data):
        status = ['COMPLETE','REJECTED','CANCELLED']
        if data['status'] == status[0] and data['transaction_type'] == 'BUY':
            self.current_orders.append([data['order_id'],data['instrument_token'],data['average_price'],data['filled_quantity']])
        if data['status'] == status[0] and data['transaction_type'] == 'SELL':
            q = self.current_orders[0][3] - data['filled_quantity']
            if q == 0:
                self.current_orders = []
            else:
                self.current_orders[0][3] = q
    
    def remainingTimeFunction(self):
        while self.threadRemTimeRunning:
            if type(self.next_update) == datetime:
                z = self.next_update - datetime.now()
                z = int(z.total_seconds())
                z = z if z>0 else -1*z
                self.remTime = z
            sleep(0.03)
    
    def queueThreadFunction(self):
        print(f'queue thread running')
        s = 0
        while self.queueThreadRunning:
            if len(self.queue) != 0:
                for ticks in self.queue:
                    try:
                        self.retrievePrices(ticks)
                    except Exception as e:
                        raise Exception(f'{e}')
                self.queue = []
            s += 1
            sleep(0.001)
    
    def update(self):
        def flatten(matrix):
            res = []
            for i in matrix:
                for j in i:
                    res.append(j)
            return res
            
        sm = flatten(self.sub_matrix)
        self.db.insert(sm)
        self.sub_matrix= [[] for i in self.names]
        self.next_update = self.func()
        self.update_number+=1
    
    def on_ticks(self,ws, ticks):
        self.queue.append(ticks)
    
    def checkOrder(self,order_id):
        def check(order_id,orders):
            status = ['OPEN','COMPLETE','CANCELLED','REJECTED','PUT ORDER REQUEST RECEIVED','VALIDATION PENDING','OPEN PENDING','MODIFY VALIDATION PENDING','MODIFY PENDING','TRIGGER PENDING','CANCEL PENDING','AMO REQ RECEIVED']
            order_status = [order['status'] for order in orders if order['order_id']==str(order_id)]
            res = 0
            for i in status:
                x = status.index(i)
                if x > 3:
                    res = 1
                    break
            return order_status,res
        
        m = ['OPEN','COMPLETE','CANCELLED','REJECTED']
        orders = self.kite.orders()
        order_status,flag = check(order_id,orders)
        if flag == 1:
            pass # do something that raise alert 
        res = 0
        for i in order_status:
            if m.index(i) > 1:
                res = 1
        return True if res == 0 else False
    
    def retrievePrices(self,ticks):
        strat_price = 0
        flag = True
        c = 0
        for i in ticks:
            inst = i['instrument_token']
            price = i['last_price']
            try:
                volume = i['volume_traded']
            except:
                volume = None
            self.channel_data[inst] = [price,volume]
        #self.db.insert(row)
    
    def ceil_dt(self,dt,time_split=300):
        nsecs = dt.minute*60 + dt.second + dt.microsecond*1e-6  
        delta = np.ceil(nsecs / time_split) * time_split - nsecs
        return dt + timedelta(seconds=delta)

    def func(self,now=None):
        if now is None:
            now = datetime.now()
        t = self.ceil_dt(now,time_split=self.interval['value'])   # timesplit requires input in seconds
        if t>now.replace(hour=15,minute=30,second=0,microsecond=0):
            t = now.replace(hour=15,minute=30,second=0,microsecond=0)
        elif t<now.replace(hour=9,minute=15,second=0,microsecond=0):
            t = now.replace(hour=9,minute=20,second=0,microsecond=0)
        return t
    
    def on_error(self,ws,code,reason):
        m = self.ticker.is_connected()
        print(f'error ({code}) , status : {"Alive" if m is True else "Dead"}\nmessage : {reason}')
    
    def startTicker(self):
        self.ticker.connect(threaded=True)
    
    def on_reconnect(self,ws,attempted_count):
        #logging.info(f'reconnecting @ {attempted_count}')
        pass
    
    def start(self,token):
        self.ticker.unsubscribe([i['instrument_token'] for i in self.channels])
        self.channel_data = {}
        self.ticker.subscribe([token['instrument_token']])
        self.ticker.set_mode(self.ticker.MODE_FULL, [token['instrument_token']])
        self.channels = [token]
        return self.full_prices()
    
    def on_connect(self,ws, response):
        ws.subscribe([i['instrument_token'] for i in self.channels])
        ws.set_mode(ws.MODE_FULL, [i['instrument_token'] for i in self.channels])
    
    def full_prices(self):
        def f1(arr):
            x = []
            for i in arr:
                try:
                    x.append([i['date'].replace(tzinfo=None),i['open'],i['high'],i['low'],i['close'],i['volume']])
                except:
                    x.append([i['date'].replace(tzinfo=None),i['open'],i['high'],i['low'],i['close'],None])
            return x
        
        candles = []
        now = datetime.now()
        last = now.date() - timedelta(days=7)
        for channel in self.channels:
            h = self.kite.historical_data(channel['instrument_token'],last,now,self.interval['name'])
            arr = f1(h)
            candles.append(arr)
        return candles
        
    def on_close(self,ws, code, reason):
        print(f'closed connection')
        #logging.info('closed connection')
    
    def getDates(self):
        h = self.kite.historical_data(self.monitor['instrument'],self.today-timedelta(days=10),self.today,self.interval['name'])
        sorted = np.sort(np.unique([i['date'].date() for i in h]))
        return sorted
    
    def getPreviousDate(self):
        d = deepcopy(self.dates)
        x = (d == self.today).any()
        if x:
            return np.delete(d,-1)[-1]
        else:
            return d[-1]
    
    def getPastPrices(self):
        def additem(names,matrices):
            df = pd.DataFrame(columns=names)
            ind = 0
            for matrix in matrices:
                indices = [i.replace(tzinfo=None).isoformat() for i in np.array(matrix)[:,0]]
                cols = [np.array(matrix)[:,1],np.array(matrix)[:,2],np.array(matrix)[:,3],np.array(matrix)[:,4]]
                for name,col in zip(names[ind*4:(ind+1)*4],cols):
                    for index,value in zip(indices,col):
                        df.loc[index,name] = value
                ind+= 1
            return df
        
        def addtodb(df):
            for i,(ind,row) in enumerate(df.iterrows()):
                q = row.values.tolist()
                res = [i,ind]
                res.extend(q)
                yield res
        
        def f1(arr):
            x = []
            for i in arr:
                x.append([i['date'],i['open'],i['high'],i['low'],i['close']])
            return x
        
        def f2(i):
            return [i['open'],i['high'],i['low'],i['close']]
        
        def tsd(arr):
            arr = np.array([i['date'].date() for i in arr])
            n = np.sort(np.unique(arr))
            num = next((i for i, j in enumerate(arr==n[-1]) if j), None)
            return num
        
        today = datetime.now().date()
        interval = self.interval['name']
        khd = self.kite.historical_data(self.monitor['instrument'],today-timedelta(days=7),today,interval)
        dates = np.sort(np.unique([i['date'].date() for i in khd]))
        yesterday = dates[-2]
        for ind,no in enumerate(self.tokens):
            try:
                hist = self.kite.historical_data(no,yesterday,today,interval)
            except Exception as e:
                raise Exception(f'{self.tokens}{self.tokens[no]}{no}{type(self.tokens)}\nexception : {e}')
            self.matrix[ind] = f1(hist[:-1])
            self.start_index.append(tsd(hist))
            try:
                self.sub_matrix[ind] = f2(hist[-1])
            except:
                raise Exception(f'index = {ind}, length of submatrix = {len(self.sub_matrix)}\nhistory = {hist}')
            print(f'\rprevious data {ind+1}/{len(self.tokens)}',end='')
        df = additem(self.db_names,self.matrix)
        dfgen = addtodb(df)
        for gen in dfgen:
            self.db.insert(gen,true=True)
        print()
    
    
    def getTokens(self):
        c = 1
        while True:
            try:
                q = self.kite.instruments('NFO')
                break
            except Exception as e:
                print(f'\rretrying to get instruments ({c}) ({e})',end='')
                c+=1
        print()
        q = [i for i in q if i['segment']=='NFO-OPT' and i['name']==self.monitor['option_name']]
        sym = self.channels[0]
        strike = round(self.kite.ltp(sym['instrument_token'])[str(sym['instrument_token'])]['last_price'],-2)
        ex=np.array([i['expiry'] for i in q])
        print(ex.dtype,ex[0])
        ex[ex==''] = datetime.now().replace(year=2999).date()
        n = np.sort(np.unique(ex))
        start_strike = strike - 600
        end_strike = strike + 600
        for i in q:
            c1 = i['segment'] == 'NFO-OPT'
            c2 = i['name'] == sym['option_name']
            c3 = i['strike'] > start_strike
            c4 = i['strike'] < end_strike
            c5 = i['expiry'] == n[0]
            cond = c1 and c2 and c3 and c4 and c5
            if cond:
                self.tokens[i['tradingsymbol']] = {
                'instrument_token':i['instrument_token'],
                'instrument_type':i['instrument_type'],
                'lot_size':i['lot_size'],
                'tradingsymbol':i['tradingsymbol']}
    
    def setup_database(self):
        db_names = []
        for i in self.names:
            db_names.extend([f'{i}_open',f'{i}_high',f'{i}_low',f'{i}_close'])
        self.db_names = db_names
        db = Database(db_names)
        return db
    
    def get(self,val):
        nv = val * 4
        names = self.db_names[nv:nv+4]
        return self.db[names]
    
    def buy(self,qty):
        order_id = self.kite.place_order(tradingsymbol=self.channels[0]['tradingsymbol'],
                                exchange=self.kite.EXCHANGE_NFO,
                                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                                quantity=qty,
                                variety=self.kite.VARIETY_REGULAR,
                                order_type=self.kite.ORDER_TYPE_MARKET,
                                product=self.kite.PRODUCT_MIS,
                                validity=self.kite.VALIDITY_IOC)
        self.orders.append(order_id)
        print(order_id)
    
    def sell(self,qty):
        order_id = self.kite.place_order(tradingsymbol=self.channels[0]['tradingsymbol'],
                                exchange=self.kite.EXCHANGE_NFO,
                                transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                                quantity=qty,
                                variety=self.kite.VARIETY_REGULAR,
                                order_type=self.kite.ORDER_TYPE_MARKET,
                                product=self.kite.PRODUCT_MIS,
                                validity=self.kite.VALIDITY_IOC)
        self.orders.append(order_id)
        print(order_id)