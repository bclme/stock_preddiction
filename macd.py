import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
start = dt.datetime(2022, 4, 1)
end = dt.datetime.now()

df = yf.Ticker('BTC').history(start=start,  end=end)[map(str.title, ['open', 'close', 'low', 'high', 'volume'])]
#df = pd.read_csv("tsla.csv")
df.ta.macd(close='close', fast=12, slow=26, append=True)

df.columns = [x.lower() for x in df.columns]

MACD_Buy=[]
MACD_Sell=[]

position=False
risk = 0.025
for i in range(0, len(df)):     
        if df['macd_12_26_9'][i] > df['macds_12_26_9'][i] and df['macd_12_26_9'][i] < 0 and (df['macd_12_26_9'][i-1] < df['macds_12_26_9'][i-1] or df['macd_12_26_9'][i-1] == df['macds_12_26_9'][i-1]):
            MACD_Sell.append(np.nan)
            if position ==False:
                MACD_Buy.append(df['close'][i])
                position=True
            else:
                MACD_Buy.append(np.nan)
        elif df['macd_12_26_9'][i] < df['macds_12_26_9'][i] and df['macd_12_26_9'][i] > 0 and (df['macd_12_26_9'][i-1] > df['macds_12_26_9'][i-1] or df['macd_12_26_9'][i-1] == df['macds_12_26_9'][i-1]):
            MACD_Buy.append(np.nan)
            if position == True:
                MACD_Sell.append(df['close'][i])
                                    
                position=False
            else:
                MACD_Sell.append(np.nan)
        elif position == True and df['close'][i] > df['close'][i - 1] * (1 - risk):
            MACD_Sell.append(df['close'][i])
            MACD_Buy.append(np.nan)
            position = False
        elif position == True and df['close'][i] < df['close'][i - 1] * (1 - risk):
            MACD_Sell.append(df['close'][i])
            MACD_Buy.append(np.nan)
            position = False
        #elif position == True and df['close'][i] < df['close'][i - 1] * (1 - risk):
        #    MACD_Sell.append(df["close"][i])
        #    MACD_Buy.append(np.nan)
        #    position = False
        else:
            MACD_Buy.append(np.nan)
            MACD_Sell.append(np.nan)


df['MACD_Buy_Signal_price'] = MACD_Buy
df['MACD_Sell_Signal_price'] = MACD_Sell
df.to_csv('tsla_stock.csv')
macd_plot = make_subplots(rows=3, cols=1)


macd_plot.append_trace(
    go.Scatter(
        x=df.index,
        y=df['close'],
        line=dict(color='#ff9900', width=1),
        name='close',
        legendgroup='1',
    ), row=1, col=1
)


macd_plot.append_trace(
    go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    ), row=1, col=1
)


#  (%k)
macd_plot.append_trace(
    go.Scatter(
        x=df.index,
        y=df['macd_12_26_9'],
        line=dict(color='#ff9900', width=2),
        name='macd',
        legendgroup='2',
    ), row=2, col=1
)
# (%d)
macd_plot.append_trace(
    go.Scatter(
        x=df.index,
        y=df['macds_12_26_9'],
        line=dict(color='#000000', width=2),
 
        legendgroup='2',
        name='signal'
    ), row=2, col=1
)

design = np.where(df['macdh_12_26_9'] < 0, '#000', '#ff9900')

macd_plot.append_trace(
    go.Bar(
        x=df.index,
        y=df['macdh_12_26_9'],
        name='histogram',
        marker_color=design,
    ), row=2, col=1
)
macd_plot.append_trace(
    go.Scatter(
        mode = 'markers',
        x=df.index,
        y=df['MACD_Buy_Signal_price'],
        
        name='Buy Signal',
        legendgroup='1',
        marker = dict(color = 'black', size = 10),
    ), row=1, col=1
)

macd_plot.append_trace(
    go.Scatter(
        mode = 'markers',
        x=df.index,
        y=df['MACD_Sell_Signal_price'],
        
        name='Sell Signal',
        legendgroup='1',
        marker = dict(color = 'blue', size = 10),
    ), row=1, col=1
)

layout = go.Layout(
    plot_bgcolor='#efefef',
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    autosize=True,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)

macd_plot.update_layout(layout)
macd_plot.show()
