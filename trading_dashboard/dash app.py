import pandas as pd
import numpy as np
import io
import mplfinance as mpf
import yfinance as yf

df = yf.download("SPY", start="2023-06-08", end="2023-06-11", interval='5m')
df.index = pd.to_datetime(df.index)
df.index = df.index.tz_localize(None)

data = '''
Time Price
"2022-06-07 11:20:00" 412.66
"2022-06-07 12:30:00" 411.350
"2022-06-07 13:50:00" 413.290
"2022-06-07 15:00:00" 414.109
"2022-06-09 13:25:00" 409.660
"2022-06-10 09:50:00" 394.130
'''

sell_df = pd.read_csv(io.StringIO(data), delim_whitespace=True)

sell_df['Time'] = pd.to_datetime(sell_df['Time'])
sell_df.set_index('Time', inplace=True)
sell_df = sell_df.reindex(df.index, axis='index', fill_value=np.NaN)
apdict = mpf.make_addplot(sell_df['Price'], type='scatter', markersize=200, marker='^')
mpf.plot(df,
         type="candle", 
         title = "Micro E-mini S&P 500 Price",  
         style="yahoo", 
         volume=True, 
         figratio=(12.00, 5.75),
         returnfig=True,
         show_nontrading=False,
         addplot=apdict
    )