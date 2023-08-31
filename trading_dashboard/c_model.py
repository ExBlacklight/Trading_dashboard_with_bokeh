import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model as keras_load_model
from tensorflow.keras.layers import Dense,Input,Conv1D,Conv1DTranspose,Flatten,Concatenate,Reshape,GlobalAveragePooling1D
from tensorflow.keras import layers
import pandas as pd
from datetime import datetime,date,timedelta
from copy import deepcopy
import pandas_ta as ta

def c_model():
    def conv_block_1(layer):
        l = Conv1D(64,kernel_size=7,strides=1,padding='same',activation='relu') (layer)
        l = Conv1D(64,kernel_size=7,strides=1,padding='same',activation='relu') (l)
        l = layers.MaxPool1D() (l)
        l = Conv1D(128,kernel_size=7,strides=1,padding='same',activation='relu') (l)
        l = Conv1D(128,kernel_size=7,strides=1,padding='same',activation='relu') (l)
        l = layers.MaxPool1D() (l)
        l = Conv1D(256,kernel_size=7,strides=1,padding='same',activation='relu') (l)
        l = Conv1D(256,kernel_size=7,strides=1,padding='same',activation='relu') (l)
        l = layers.MaxPool1D() (l)
        o = layers.GlobalAveragePooling1D() (l)
        return o

    def conv_block_2(layer):
        l = Conv1D(32,kernel_size=7,strides=1,padding='same',activation='relu') (layer)
        l = Conv1D(32,kernel_size=7,strides=1,padding='same',activation='relu') (l)
        l = layers.MaxPool1D() (l)
        l = Conv1D(64,kernel_size=7,strides=1,padding='same',activation='relu') (l)
        l = Conv1D(64,kernel_size=7,strides=1,padding='same',activation='relu') (l)
        l = layers.GlobalAveragePooling1D() (l)
        o = Reshape((1,64)) (l)
        return o

    inp1 = Input(shape=(50,14))
    inp2= Input(shape=(2,11))
    l1 = conv_block_1(inp1)
    l2 = conv_block_2(inp2)
    l1 = Dense(200,activation='relu') (l1)
    l1 = Dense(200,activation='relu') (l1)
    l1 = Dense(100,activation='relu') (l1)
    l1 = Dense(100,activation='relu') (l1)
    l1 = Dense(64,activation='relu') (l1)
    l1 = layers.Reshape((1,64)) (l1)
    l = layers.Concatenate(axis=-2) ([l1,l2])
    l = Conv1DTranspose(50,kernel_size=16,strides=1,padding='same',activation='relu') (l)
    l = Conv1DTranspose(40,kernel_size=16,strides=1,padding='same',activation='relu') (l)
    l = Conv1DTranspose(30,kernel_size=16,strides=2,padding='same',activation='relu') (l)
    l = Conv1DTranspose(20,kernel_size=16,strides=1,padding='same',activation='relu') (l)
    out = Conv1DTranspose(11,kernel_size=16,strides=1,padding='same',activation='relu') (l)
    model = Model(inputs=[inp1,inp2],outputs=[out])
    model.compile(loss='mse',optimizer='adam')
    return model

def minimum_functions(df):
    df = deepcopy(df)
    try:
        df['vwap'] = calculate_vwap(df)
    except:
        df['vwap'] = [np.nan for i in range(len(df))]
    df = df.set_index(pd.DatetimeIndex(df['date']))
    df['ema_20'] = ta.ema(df['close'],length=20)
    df['ema_50'] = ta.ema(df['close'],length=50)
    df['ema_100'] = ta.ema(df['close'],length=100)
    return df[['open','high','low','close','vwap','ema_20','ema_50','ema_100']]

def all_functions(df):
    df = df.dropna()
    df = deepcopy(df)
    df['vwap'] = calculate_vwap(df)
    df = df.set_index(pd.DatetimeIndex(df['date']))
    df['ema_20'] = ta.ema(df['close'],length=20)
    df['ema_50'] = ta.ema(df['close'],length=50)
    df['ema_100'] = ta.ema(df['close'],length=100)
    df['vwap_p1'] = df['vwap'] * (1 + 0.1)
    df['vwap_n1'] = df['vwap'] * (1 - 0.1)
    df['vwap_p2'] = df['vwap'] * (1 + 0.2)
    df['vwap_n2'] = df['vwap'] * (1 - 0.2)
    df['vwap_p3'] = df['vwap'] * (1 + 0.3)
    df['vwap_n3'] = df['vwap'] * (1 - 0.3)
    df = df.dropna()
    return df[['open','high','low','close','ema_20','ema_50','ema_100','vwap','vwap_p1','vwap_n1','vwap_p2','vwap_n2','vwap_p3','vwap_n3','date']]

def big_num(val):
    def convert_to_binary(number):
        binary = bin(number)[2:]
        padded_binary = binary.zfill(5)
        return np.array(list(padded_binary),dtype=np.float32)

    def conversion(val):
        dec = np.round(val%1,decimals=3)
        n = 32
        q = int(val)//n
        d = int(val)%n
        q=convert_to_binary(q)
        d=convert_to_binary(d)
        return np.concatenate([q,d,[dec]])

    if isinstance(val, (int, float, complex)):
        return conversion(val).astype(np.float32)
    elif isinstance(val,(np.ndarray,list)):
        res = []
        for v in val:
            res.append(conversion(v).tolist())
        try:
            return np.array(res,dtype=np.float32)
        except Exception as e:
            raise Exception(f'{res}\n{e}')
    else:
        raise Exception(f'need to be an array or a number')

def convert_back(val):
    def rounding(val):
        val[np.arange(11)<10] = np.round(val[:10])
        val[val>1]=1
        return val

    def convert_to_decimal(binary):
        binary_str = ''.join([str(int(bit)) for bit in binary])
        try:
            decimal = int(binary_str, 2)
        except Exception as e:
            raise Exception(f'{binary_str}\n{e}')
        return decimal

    def conversion_back(val):
        val = rounding(val)
        q_binary = val[:5]
        d_binary = val[5:10]
        dec = val[10]
        q_decimal = convert_to_decimal(q_binary)
        d_decimal = convert_to_decimal(d_binary)
        original_val = q_decimal * 32 + d_decimal + dec
        return original_val

    if isinstance(val, (int, float, complex)):
        return conversion_back(val)
    elif isinstance(val, (np.ndarray, list)):
        res = []
        for v in val:
            res.append(conversion_back(v))
        return np.array(res)
    else:
        raise Exception('Value needs to be an array or a number.')

def encode(arr):
    maxi = np.max(arr)
    mini = np.min(arr)
    k = (arr - mini)/(maxi-mini)
    ar2 = np.array([[big_num(maxi),big_num(mini)]])
    return k[None,...].astype(np.float32),ar2.astype(np.float32)

def make_df(data):
    df = pd.DataFrame(data)
    df = df[['open','high','low','close','volume']]
    try:
        df['date'] = np.array([datetime.strptime(i,'%Y-%m-%d %H:%M:%S').date() for i in data['time']])
    except:
        df['date'] = np.array([datetime.strptime(i,'%Y-%m-%d %H:%M').date() for i in data['time']])
    df = all_functions(df)
    return df

def simple_df(data):
    df = pd.DataFrame(data)
    df = df[['open','high','low','close','volume']]
    try:
        df['date'] = np.array([datetime.strptime(i,'%Y-%m-%d %H:%M:%S').date() for i in data['time']])
    except:
        df['date'] = np.array([datetime.strptime(i,'%Y-%m-%d %H:%M').date() for i in data['time']])
    df = minimum_functions(df)
    return df[['vwap','ema_20','ema_50','ema_100']]

def patch_indicators(data):
    df = pd.DataFrame(data)
    df = df[['open','high','low','close','volume']]
    try:
        df['date'] = np.array([datetime.strptime(i,'%Y-%m-%d %H:%M:%S').date() for i in data['time']])
    except:
        df['date'] = np.array([datetime.strptime(i,'%Y-%m-%d %H:%M').date() for i in data['time']])
    df = minimum_functions(df)
    return df[['vwap','ema_20','ema_50','ema_100']]

def full_convert(data):
    data = deepcopy(data)
    df = make_df(data)
    x1,x2 = encode(df[['open','high','low','close','ema_20','ema_50','ema_100','vwap','vwap_p1','vwap_n1','vwap_p2','vwap_n2','vwap_p3','vwap_n3']].values[-50:])
    return x1,x2

def calculate_vwap(df):
    vwap_values = []
    grouped = df.groupby('date')
    for _, group in grouped:
        typical_price = (group['high'] + group['low'] + group['close']) / 3
        total_volume = group['volume'].cumsum()
        cumulative_typical_price_volume = (typical_price * group['volume']).cumsum()
        vwap = cumulative_typical_price_volume / total_volume
        vwap_values.extend(vwap)
    return vwap_values
