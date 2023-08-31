import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from bokeh.layouts import column,row
from bokeh.models.widgets import Button,RadioButtonGroup
from bokeh.models import Dropdown,HoverTool,Div,Span
from random import randint
import datetime
from functools import partial
from KT import KT
import numpy as np
from datetime import datetime,time,timedelta
from time import sleep
import threading
from c_model import c_model,full_convert,convert_back,simple_df,patch_indicators
from tensorflow.keras.models import load_model as keras_load_model
from copy import deepcopy
curdoc().theme = 'dark_minimal'

# Function to generate random candlestick data
def generate_candlestick_data():
    global xaxis,x2axis
    arr = kt.full_prices()[0]
    xaxis = [i for i in range(len(arr))]
    x2axis= [xaxis[-1]+1]
    time = [str(i[0]) for i in arr]
    op = [i[1] for i in arr]
    hi = [i[2] for i in arr]
    lo = [i[3] for i in arr]
    cl = [i[4] for i in arr]
    vol = [i[5] for i in arr]
    d = [i-j for i,j in zip(cl,op)]
    color = [(0,255,0,1) if i>0 else (255,0,0,1) for i in d]
    return {'x':xaxis,'open':op,'high':hi,'low':lo,'close':cl, 'color':color, 'time':time, 'volume':vol}

def ceil_dt(dt,time_split=300):
    nsecs = dt.minute*60 + dt.second + dt.microsecond*1e-6  
    delta = np.ceil(nsecs / time_split) * time_split - nsecs
    return dt + timedelta(seconds=delta)


def func(now=None):
    if now is None:
        now = datetime.now()
    t = ceil_dt(now,time_split=kt.interval['value'])   # timesplit requires input in seconds
    if t>now.replace(hour=15,minute=30,second=0,microsecond=0):
        t = now.replace(hour=15,minute=30,second=0,microsecond=0)
    elif t<now.replace(hour=9,minute=15,second=0,microsecond=0):
        t = now.replace(hour=9,minute=20,second=0,microsecond=0)
    return t

def indicator_update():
    global data2
    df = patch_indicators(data)
    row = df.iloc[-1]
    patch = {}
    patch['vwap'] = [(xaxis[-1],row['vwap'])]
    patch['ema_20'] = [(xaxis[-1],row['ema_20'])]
    patch['ema_50'] = [(xaxis[-1],row['ema_50'])]
    patch['ema_100'] = [(xaxis[-1],row['ema_100'])]
    data2['vwap'][-1] = row['vwap']
    data2['ema_20'][-1] = row['ema_20']
    data2['ema_50'][-1] = row['ema_50']
    data2['ema_100'][-1] = row['ema_100']
    source3.patch(patch)
    

def update():
    global update_time,xaxis,data,counter,x2axis,buy_multiplier,data2,volume_minus_one,cvol,pvol,rem_details_text,rem_details_text_2
    try:
        price = kt.channel_data[inst][0]
        volume = kt.channel_data[inst][1]
        if volume_minus_one == None:
            volume_minus_one = volume
        else:
            cvol += (volume - volume_minus_one)
        volume_minus_one = volume
        try:
            buy_multiplier = int(budget/(lot_size*4*kt.channel_data[inst][0]))
        except:
            buy_multiplier = 0
    except:
        return 
    try:
        buy_price = np.round(kt.current_orders[0][2],decimals=2)
        qty = np.round(kt.current_orders[0][3],decimals=2)
    except:
        buy_price = 0
        qty = 0
    rem_details_text.text=f'<b>current_price</b> = {price}  <b>current_qty</b> = {qty}'
    rem_details_text_2.text=f'<b>current_trade_price</b> = {buy_price} <b>open positions</b> = {radio_lots}'
    now = datetime.now().time()
    if datetime.now() > update_time:# major update
        update_time = func()
        x = xaxis[-1]
        patch = {'open':[price],'high':[price],'low':[price],'close':[price],'x':[x+1],'color':['black'],'time':[str(update_time)],'volume':[cvol]}
        p2 = {'open':[price],'high':[price],'low':[price],'close':[price],'x':[x+2],'color':['white']}
        if now > time(9,14,55) and now < time(15,30,5):
            source.stream(patch,rollover=None)
            source2.stream(p2,rollover=1)
        xaxis.append(x+1)
        x2axis = [xaxis[-1]+1]
        data['open'].append(price)
        data['high'].append(price)
        data['low'].append(price)
        data['close'].append(price)
        data['color'].append('black')
        data['time'].append(str(update_time))
        data['volume'].append(cvol)
        df = patch_indicators(data)
        row = df.iloc[-1][['vwap','ema_20','ema_50','ema_100']].values.tolist()
        x = data2['x'][-1]
        patch = {'x':[x+1],'vwap':[row[0]],'ema_20':[row[1]],'ema_50':[row[2]],'ema_100':[row[3]]}
        print(patch)
        if now > time(9,14,55) and now < time(15,30,5):
            source3.stream(patch,rollover=None)
        data2['x'].append(x+1)
        data2['vwap'].append(row[0])
        data2['ema_20'].append(row[1])
        data2['ema_50'].append(row[2])
        data2['ema_100'].append(row[3])
        print(len(data2['x']),len(data2['vwap']))
        pvol = 0
        volume_minus_one = None
        cvol = 0
        print('major_update')
    else:# mini update
        patch = {}
        if price > data['high'][-1]:
            patch['high'] = [(xaxis[-1],price)]
            data['high'][-1] = price
        if price < data['low'][-1]:
            patch['low'] = [(xaxis[-1],price)]
            data['low'][-1] = price
        if price != data['close'][-1]:
            patch['close'] = [(xaxis[-1],price)]
            data['close'][-1] = price
        d = data['close'][-1] - data['open'][-1]
        if d > 0:
            patch['color'] = [(xaxis[-1],(0,255,0,1))]
            data['color'][-1] = (0,255,0,1)
        else:
            patch['color'] = [(xaxis[-1],(255,0,0,1))]
            data['color'][-1] = (255,0,0,1)
        if patch != {} and now > time(9,14,55) and now < time(15,30,5):
            source.patch(patch)
            data['volume'][-1] = cvol
        df = patch_indicators(data)
        row = df.iloc[-1]
        patch = {}
        patch['vwap'] = [(data2['x'][-1],row['vwap'])]
        patch['ema_20'] = [(data2['x'][-1],row['ema_20'])]
        patch['ema_50'] = [(data2['x'][-1],row['ema_50'])]
        patch['ema_100'] = [(data2['x'][-1],row['ema_100'])]
        data2['vwap'][-1] = row['vwap']
        data2['ema_20'][-1] = row['ema_20']
        data2['ema_50'][-1] = row['ema_50']
        data2['ema_100'][-1] = row['ema_100']
        if now > time(9,14,55) and now < time(15,30,5):
            source3.patch(patch)

def predict():
    if data['volume'][-1] is None:
        return -1
    try:
        x1,x2 = full_convert(data)
    except ValueError as e:
        return -1
    pred = model([x1,x2]).numpy()
    op,hi,lo,cl = convert_back(pred[0])
    color = (0,255,0,0.5) if cl-op > 0 else (255,0,0,0.5)
    #color = 'blue' if cl-op > 0 else 'black'
    patch = {}
    patch['open'] = [(0,op)]
    patch['high'] = [(0,hi)]
    patch['low'] = [(0,lo)]
    patch['close'] = [(0,cl)]
    patch['color'] = [(0,color)]
    try:
        source2.patch(patch)
    except Exception as e:
        raise Exception(f'{source2.data}\n{e}')

def data_to_dict(data):
    new_data = {}
    data = np.array(data)
    new_data['time'] = [str(i) for i in data[:,0].tolist()]
    new_data['open'] = data[:,1].tolist()
    new_data['high'] = data[:,2].tolist()
    new_data['low'] = data[:,3].tolist()
    new_data['close'] = data[:,4].tolist()
    new_data['volume'] = data[:,5].tolist()
    new_data['x'] = np.arange(len(new_data['open'])).tolist()
    new_data['color'] = [(0,255,0,1) if i-j > 0 else (255,0,0,1) for i,j in zip(new_data['close'],new_data['open'])]
    return new_data


def handler(event):
    global xaxis,data,source,inst,x2axis,source2,p,lot_size,data2,source3,pvol,cvol,volume_minus_one
    new_data = kt.start(kt.tokens[event.item])
    new_data = data_to_dict(new_data[0])
    pvol = new_data['volume'][-1]
    cvol = pvol
    volume_minus_one = None
    data2 = simple_df(new_data).to_dict('list')
    xaxis = new_data['x']
    x = np.arange(len(data2['vwap'])).tolist()
    data2['x'] = x
    x2axis = new_data['x'][-1] + 1
    data = new_data
    source.data = new_data
    source2.data = {'x':[x2axis],'open':[data['close'][-1]],'high':[data['close'][-1]],'low':[data['close'][-1]],'close':[data['close'][-1]], 'color':['white']}
    source3.data = data2
    inst = kt.channels[0]['instrument_token']
    title = kt.channels[0]['tradingsymbol']
    lot_size = kt.channels[0]['lot_size']
    try:
        buy_multiplier = int(budget/(lot_size*4*kt.channel_data[inst][0]))
    except:
        buy_multiplier = 0
    p.title.text = f'{title[-7:-2]} {title[-2:]}'

def buy_button_handler(event):
    global radio_lots,sell_multiplier
    radio_lots = buy_sheet[int(radio_button_group.active)]
    sell_multiplier = deepcopy(buy_multiplier)
    shares_bought = buy_multiplier*radio_lots * lot_size
    print(shares_bought,buy_multiplier)
    kt.buy(shares_bought)

def sell_button_handler(event):
    global radio_lots,sell_multiplier
    size = buy_sheet[int(radio_button_group.active)]
    shares_selling = size * lot_size * sell_multiplier
    if size > radio_lots:
        kt.sell(sell_multiplier*radio_lots * lot_size)
        radio_lots = 0
    else:
        kt.sell(shares_selling)
        radio_lots -= size

def details_update():
    global profit_text,profit
    profit_text.text = f'profit = {np.round(getProfit(),decimals=2)}'
    

def getProfit():
    t = kt.kite.positions()['day']
    tokens = [i['instrument_token'] for i in t]
    if len(tokens) == 0:
        return 0
    ltps = kt.kite.ltp(tokens)
    prices = [ltps[f'{i}']['last_price'] for i in tokens]
    p = 0
    for i,price in zip(t,prices):
        p += (i['sell_value'] - i['buy_value']) + (i['quantity'] * price * i['multiplier'])
    return p

def radio_button_update():
    radio_button_group.active

kt = KT()
profit = getProfit()
radio_labels = ['1', '2','3','4']
budget = kt.kite.margins()['equity']['available']['live_balance']
buy_sheet = np.array([1,2,3,4])
buy_multiplier = 1
pvol = None
cvol = 0
volume_minus_one = None
try:
    lot_size = kt.channels[0]['lot_size']
except:
    lot_size = 0
xaxis = []
x2axis = []
data = generate_candlestick_data()
data2 = simple_df(data).to_dict('list')
x = np.arange(len(data2['vwap'])).tolist()
data2['x'] = x
source = ColumnDataSource(data)
source2 = ColumnDataSource({'x':x2axis,'open':[data['close'][-1]],'high':[data['close'][-1]],'low':[data['close'][-1]],'close':[data['close'][-1]], 'color':['white']})
source3 = ColumnDataSource({'x':xaxis,'vwap':data2['vwap'],'ema_20':data2['ema_20'],'ema_50':data2['ema_50'],'ema_100':data2['ema_100']})
model = c_model()
model.load_weights(f'model_2_weights.h5')
counter= 30
update_time = func()
inst = kt.channels[0]['instrument_token']
radio_lots = 0
# Create the candlestick figure
hover = HoverTool()
hover.tooltips= [
    ('time','@time'),
    ('open','@open'),
    ('high','@high'),
    ('low','@low'),
    ('close','@close'),
]
#hover.formatters = {'@DateTime':'datetime'}
#Tooltips = [('x','@x'),()]

p = figure(title="Bank Nifty", x_axis_type='linear', height=500,width=800,tools="pan,wheel_zoom,reset")


p.segment(x0='x', y0='high', x1='x', y1='low', color="color", source=source)
rend1 = p.vbar(x='x', width=0.5, top='open', bottom='close', fill_color="color", line_color="color", source=source)

p.line(x='x', y='vwap', line_width=2, line_color='white',source=source3)
p.line(x='x', y='ema_20', line_width=1, line_color='green',source=source3)
p.line(x='x', y='ema_50', line_width=1.5, line_color='yellow',source=source3)
p.line(x='x', y='ema_100', line_width=2, line_color='red',source=source3)

p.segment(x0='x', y0='high', x1='x', y1='low', color="color", source=source2)
rend2 = p.vbar(x='x', width=0.5, top='open', bottom='close', fill_color="color", line_color="color", source=source2)
hover.mode='vline'
hover.renderers = [rend1,rend2]
p.add_tools(hover)

menu = [(f'_CE_{i[-7:-2]} : {i}' if i[-2:]=="CE" else f'__PE__{i[-7:-2]} : {i}',i) for ind,i in enumerate(kt.tokens.keys())]

dropdown = Dropdown(label="all Strikes", button_type="warning", menu=menu, height= 70, width=640,styles={'font-size': '130%', 'color': '#ff0000'})
dropdown.on_click(handler)

buy_button = Button(label="BUY", button_type="success",height=50,width=100,align="center",styles={'font-size': '100%', 'color': '#ff0000'})
sell_button = Button(label="SELL", button_type="success",height=50,width=100,align="center",styles={'font-size': '100%', 'color': '#ff0000'})

buy_button.on_click(buy_button_handler)
sell_button.on_click(sell_button_handler)

radio_button_group = RadioButtonGroup(labels=radio_labels, active=0,height=70,width=320)
radio_button_group.on_change('active', lambda attr, old, new: radio_button_update())

# Create the layout and add the figure and button
profit_text= Div(text=f'Profit = {np.round(profit,decimals=2)}',styles={'font-size': '100%', 'color': '#ffffff'},align="center")
rem_details_text = Div(text=f'<b>current_price</b> = None  <b>current_qty</b> = None', styles={'font-size': '100%', 'color': '#ffffff'},align="center")
rem_details_text_2 = Div(text=f'<b>current_trade_price</b> = None <b>open positions</b> = None ', styles={'font-size': '100%', 'color': '#ffffff'},align="center")
details_area = row(children=[profit_text],align="center")
side_area = column(children=[details_area,rem_details_text,rem_details_text_2,radio_button_group,buy_button,sell_button])
graph_area = row(children=[p,side_area])
layout = column(children=[dropdown,graph_area],background='#1a1a1a', sizing_mode='stretch_both')

# Add the layout to the current document
curdoc().add_root(layout)
curdoc().add_periodic_callback(update,200)
curdoc().add_periodic_callback(predict,3000)
curdoc().add_periodic_callback(details_update,1000)

