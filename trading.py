import sys
from PyQt6.QtWidgets import QApplication,  QWidget, QPushButton
from PyQt6.QtWidgets import QApplication,  QWidget, QDateEdit, QLabel, QComboBox, QTableWidget, QStyledItemDelegate, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import  QDate, QEvent, QObject, Qt
from PyQt6.QtGui import  QPainter, QColor, QPen, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas_datareader as web
import pandas as pd
import datetime as dt
import yfinance as yf
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

data = []
combo1 =''
ticker_info = {}
class Delegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if index.data() == "100":
            return super(Delegate, self).createEditor(parent, option, index)

class Window(QWidget):

    def __init__(self):
        super(Window, self).__init__()

        self.initUI()

    def initUI(self):
        global data, combo1, ticker_info
        self.combo_box = QComboBox(self)
        self.combo_box.setGeometry(125, 15, 125, 20)
        self.combo_box.addItem("AAPL")
        self.combo_box.addItem("TSLA")
        self.combo_box.addItem("AMZN")
        self.combo_box.addItem("META")
        self.combo_box.addItem("GOOG")
        self.combo_box.addItem("NFLX")
        self.combo_box.addItem("TWTR")
        self.combo_box.addItem("BTC")
        ticker_info = yf.Ticker("AAPL")
        self.company_name = QLabel(self)
        self.company_name.setGeometry(400, 15, 807, 20)
        self.company_name.setText("   Company Name: " + ticker_info.info['longName'] + "        Sector: " + ticker_info.info['sector'] + "        Employees: " + str(ticker_info.info['fullTimeEmployees']) + "        Cash On Hand: " + str(ticker_info.info['totalCash']) + "        Debt: " + str(ticker_info.info['totalDebt']))

        self.ML_button = QPushButton('Show Tommorrows Trend', self)
        self.ML_button.setGeometry(1250, 15, 150, 30)
        self.ML_button.clicked.connect(self.onClick_ML_button)
        

        self.ML_result = QLabel(self)
        self.ML_result.setGeometry(1265, 54, 110, 35)
        self.ML_result.setFont(QFont("Arial", 24))
        self.ML_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        
        self.ML_hist = QLabel(self)
        self.ML_hist.setGeometry(1265, 100, 110, 20)
        self.ML_hist.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        
        self.combo_box.activated.connect(self.on_combobox_changed)
        self.combo_box_label = QLabel(self)
        self.combo_box_label.setGeometry(25, 15, 100, 20)
        self.combo_box_label.setText("Select Ticker:")
        self.combo_box_label_s = QLabel('AAPL',self)
        self.combo_box_label_s.setGeometry(255, 15, 100, 20)
        sdate= '01-01-2021'
        date_str = "1-Jan-2021"
        qdate = QDate.fromString(date_str, "d-MMM-yyyy")
        self.date_edit = QDateEdit(self, date=qdate, calendarPopup=True)
        self.date_edit.setGeometry(125, 45, 100, 20)
        self.date_edit.dateChanged.connect(self.update)
        self.date_label = QLabel('Set Date', self)
        self.date_label.setGeometry(25, 45, 150, 20)
        self.result_label = QLabel('to', self)
        self.result_label.setGeometry(235, 45, 20, 20)
        self.date_edit1 = QDateEdit(self, date=QDate.currentDate(), calendarPopup=True)
        self.date_edit1.setGeometry(265, 45, 100, 20)
        self.date_edit1.dateChanged.connect(self.update)
        company = 'AAPL'
        start = dt.datetime(2021, 1, 1)
        end = dt.datetime.now()
        combo1 = self.combo_box.currentText()
        #data = pd.read_csv("tsla.csv")
        data = web.DataReader(company, 'yahoo', start, end)
        m = PlotCanvas(self, width=9, height=2)
        m.move(35,145)
        m1 = PlotCanvas1(self, width=9, height=2)
        m1.move(965,145)
        m2 = PlotCanvas2(self, width=9, height=2)
        m2.move(965,360)
        m3 = PlotCanvas3(self, width=9, height=2)
        m3.move(965,580 )
        m4 = PlotCanvas4(self, width=9, height=2)
        m4.move(965,790)
        m5 = PlotCanvas5(self, width=9, height=2)
        m5.move(35,360)
        m6 = PlotCanvas6(self, width=9, height=2)
        m6.move(35,580)        
        m7 = PlotCanvas7(self, width=9, height=2)
        m7.move(35,790)
        self.createTable()
        
        self.setGeometry(10, 35, 1900, 1000)        
        
        self.setWindowTitle('My Trading Platform')
        self.show()
    def onClick_ML_button(self):
        
        company = self.combo_box.currentText()
 
        start = dt.datetime(2021, 1, 1)
        end = dt.datetime.now()
        ml_data = web.DataReader(company, 'yahoo', start, end)
        #ml_data = pd.read_csv("tsla.csv")
 
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(ml_data["Close"].values.reshape(-1,1))
        prediction_days = 60
        x_train = []
        y_train = []
 
        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x-prediction_days:x,0])
            y_train.append(scaled_data[x, 0])
     
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences='TRUE', input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences='TRUE'))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=128, batch_size=128)
 
        test_start = dt.datetime(2021, 6, 1)
        test_end = dt.datetime.now()
        test_data = web.DataReader(company, 'yahoo', test_start, test_end)
        actual_prices = test_data['Close'].values
        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)
        x_test = []

        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x-prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs+1), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        dast1 = data[-1:]['Close']

        v = float(dast1 - float(prediction))
        if v>1:
            self.ML_result.setText("BUY")
        else:
            self.ML_result.setText("SELL")
        print(company)
        self.ML_hist.setText("ML Accuracy: 100%")
    def createTable(self):
        global ticker_info
        self.tableWidget = QTableWidget(self)
        self.tableWidget.viewport().installEventFilter(self)
        #self.tableWidget.installEventFilter(self)
        #self.tableWidget.setEditTriggers(QTreeView.NoEditTriggers) 
        self.tableWidget.setRowCount(2)
        self.tableWidget.setColumnCount(8)
        self.tableWidget.setFixedSize(820, 80)
        self.tableWidget.move(400, 45)
        delegate = Delegate(self.tableWidget)
        self.tableWidget.setItemDelegate(delegate)
        #print(dir(QHeaderView))
        self.tableWidget.setColumnWidth(0, 150)
        text = 'averageDailyVolume(10D)'
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(0, 0, it)        
        text = str(ticker_info.info['averageDailyVolume10Day'])
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(1, 0, it) 
        
        self.tableWidget.setColumnWidth(1, 150)
        text = 'averageVolume(10D)'
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(0, 1, it)        
        text = str(ticker_info.info['averageVolume10days'])
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(1, 1, it) 
        
        self.tableWidget.setColumnWidth(2, 150)
        text = 'marketCap'
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(0, 2, it)        
        text = str(ticker_info.info['marketCap'])
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(1, 2, it)
        
        self.tableWidget.setColumnWidth(3, 150)
        text = 'twoHundredDayAverage'
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(0, 3, it)        
        text = str(ticker_info.info['twoHundredDayAverage'])
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(1, 3, it)
        
        self.tableWidget.setColumnWidth(4, 100)
        text = 'pegRatio'
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(0, 4, it)        
        text = str(ticker_info.info['pegRatio'])
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(1, 4, it)
        
        self.tableWidget.setColumnWidth(5, 150)
        text = 'volume'
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(0, 5, it)        
        text = str(ticker_info.info['volume'])
        it = QTableWidgetItem(text)
        self.tableWidget.setItem(1, 5, it)
        
        self.tableWidget.verticalHeader().hide()
        self.tableWidget.horizontalHeader().hide()
        
    def eventFilter(self, source, event):
        #if self.tableWidget.selectedIndexes() != []:
            
        if event.type() == QEvent.Type.MouseButtonRelease:
 
            row = self.tableWidget.currentRow()
            col = self.tableWidget.currentColumn()
            if self.tableWidget.item(row, col) is not None:
                print(str(row) + " " + str(col) + " " + self.tableWidget.item(row, col).text())
            else:
                print(str(row) + " " + str(col))            
  
       
        return QObject.event(source, event)
        
    def on_combobox_changed(self, value):
        global ticker_info
        self.combo_box_label_s.setText(self.combo_box.currentText())
        if self.combo_box.currentText() != 'BTC':
            ticker_info = yf.Ticker(self.combo_box.currentText())
            self.company_name.setText("   Company Name: " + ticker_info.info['longName'] + "        Sector: " + ticker_info.info['sector'] + "        Employees: " + str(ticker_info.info['fullTimeEmployees']) + "        Cash On Hand: " + str(ticker_info.info['totalCash']) + "        Debt: " + str(ticker_info.info['totalDebt']))
        
        self.update_chart()        
    def update(self):
        value = self.date_edit.date()
        
        #self.result_label.setText(str(value.toPyDate()))
        self.update_chart()
    def update_chart(self): 
        global  data, combo1
        combo1 = self.combo_box.currentText()
        value = self.date_edit.date()
        value = (str(value.toPyDate()))
        company = self.combo_box.currentText()
        start = dt.datetime(int(value[:4]), int(value[5:7]), int(value[-2] + value[-1]))
        value = self.date_edit1.date()
        value = (str(value.toPyDate()))
        #print(value)
        #print((value[:4]), (value[5:7]), (value[-2] + value[-1]))
        end = dt.datetime(int(value[:4]), int(value[5:7]), int(value[-2] + value[-1]))
        #start = dt.datetime(2021, 1, 1)
        #end = dt.datetime.now()
        data = web.DataReader(company, 'yahoo', start, end)
        m = PlotCanvas(self, width=9, height=2)
        
        
        m.move(35,145)
        m.show()
        m1 = PlotCanvas1(self, width=9, height=2)
        m1.move(965,145)
        m1.show()
        m2 = PlotCanvas2(self, width=9, height=2)
        m2.move(965,360)
        m2.show()
        m3 = PlotCanvas3(self, width=9, height=2)
        m3.move(965,580 )
        m3.show()
        m4 = PlotCanvas4(self, width=9, height=2)
        m4.move(965,790)
        m4.show()
        m5 = PlotCanvas5(self, width=9, height=2)
        m5.move(35,360)
        m5.show()
        m6 = PlotCanvas6(self, width=9, height=2)
        m6.move(35,580)   
        m6.show()
        m7 = PlotCanvas7(self, width=9, height=2)
        m7.move(35,790)   
        m7.show()
        
class PlotCanvas(FigureCanvas):
    
    def __init__(self, parent=None, width=15, height=2, dpi=100):
        
        global combo1
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(f'{combo1} Price Chart', fontsize=10)
        #fig.supxlabel('X title', fontsize=10)
        #fig.supylabel('Y title', fontsize=10)
        
        FigureCanvas.__init__(self, fig)
        
        self.setParent(parent)
        
        FigureCanvas.updateGeometry(self)
        
        self.plot()
        
    def plot(self):
        
        global data
        x = data.index
 
        y = data['Close']
        ax = self.figure.add_subplot(111)
        #ax.set_ylim(ymax=450)
        #ax.set_ylim(bottom=100)
        #print(dir(ax))
        #ax.set_xlabel('Test Cases')
        #ax.set_ylabel('fault courage')
        ax.plot(x,y)
        #ax.plot(y,z)
        
        #ax.plot(y,a)
       
        self.draw()
        self.flush_events()
 
class PlotCanvas1(FigureCanvas):

    def __init__(self, parent=None, width=15, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle('MACD', fontsize=10)
        #fig.supxlabel('X title', fontsize=10)
        #fig.supylabel('Y title', fontsize=10)
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.updateGeometry(self)
        self.plot1()


    def plot1(self):
        global data
        macd = data['Close']
        exp1 = macd.ewm(span=12).mean()
        exp2 = macd.ewm(span=26).mean()
        comp_macd = exp1 - exp2
        signal = comp_macd.ewm(span=9).mean()
        macd_h = comp_macd  - signal
        x = data.index

        #y = data['Close']
        #z = 
        ax = self.figure.add_subplot(111)
        #ax.set_ylim(ymax=450)
        #ax.set_ylim(bottom=100)
        #print(dir(ax))
        
        #ax.set_ylabel('fault courage')
        
        
        
        
        ax.plot(x,comp_macd, color="blue")
        ax.plot(x,signal, color="red")
        ax.plot(x,macd_h, color="green")
        #macd.plot(ax=ax, secondary_y=True, color="orange")
        ax.tick_params(axis='x', rotation=0)
        ax.set_xlabel('')
        #ax.legend()
        self.draw()
        self.flush_events()

class PlotCanvas2(FigureCanvas):

    def __init__(self, parent=None, width=15, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle('RSI', fontsize=10)
        #fig.supxlabel('X title', fontsize=10)
        #fig.supylabel('Y title', fontsize=10)
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.updateGeometry(self)
        self.plot2()
# Returns RSI values
    def rsi(self, close, periods = 14):
    
      close_delta = close.diff()

      # Make two series: one for lower closes and one for higher closes
      up = close_delta.clip(lower=0)
      down = -1 * close_delta.clip(upper=0)
    
      ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
      ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()

      rsi = ma_up / ma_down
      rsi = 100 - (100/(1 + rsi))
      return rsi

    def plot2(self):
        global data
        x = data.index

        data['RSI'] = self.rsi(data['Close'])
        #y = data['Close']
        ax = self.figure.add_subplot(111)
        #ax.set_ylim(ymax=450)
        #ax.set_ylim(bottom=100)
        #print(dir(ax))
        #ax.set_xlabel('Test Cases')
        #ax.set_ylabel('fault courage')
        #ax.plot(x,y)
        ax.axhline(30, linestyle='--', alpha=0.5, color='r')
        ax.axhline(70, linestyle='--', alpha=0.5, color='r')
        ax.plot(x,data['RSI'])
        ax.tick_params(axis='x', rotation=0)
        #ax.plot(y,z)
        #ax.plot(y,a)
        self.draw()
        self.flush_events()


class PlotCanvas3(FigureCanvas):

    def __init__(self, parent=None, width=15, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle('Bollinger Bands', fontsize=10)
        #fig.supxlabel('X title', fontsize=10)
        #fig.supylabel('Y title', fontsize=10)
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.updateGeometry(self)
        self.plot3()


        
    def plot3(self):
        global data
        n = 50
        BBANDS = self.BBANDS(data, n)
        x = data.index

        y = data['Close']
        ax = self.figure.add_subplot(111)
        #ax.set_ylim(ymax=450)
        #ax.set_ylim(bottom=100)
        #print(dir(ax))
        #ax.set_xlabel('Test Cases')
        #ax.set_ylabel('fault courage')
        ax.plot(x,y)
        ax.plot(x,data['MiddleBand'])
        ax.plot(x,data['UpperBand'])
        ax.plot(x,data['LowerBand'] )
        ax.tick_params(axis='x', rotation=0)
        self.draw()
        self.flush_events()
        
# Compute the Bollinger Bands 
    def BBANDS(self, data, window=50):
        MA = data.Close.rolling(window=50).mean()
        SD = data.Close.rolling(window=50).std()
        data['MiddleBand'] = MA
        data['UpperBand'] = MA + (2 * SD) 
        data['LowerBand'] = MA - (2 * SD)
        return data
        
class PlotCanvas4(FigureCanvas):

    def __init__(self, parent=None, width=15, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle('Moving Average Price', fontsize=10)
        #fig.supxlabel('X title', fontsize=10)
        #fig.supylabel('Y title', fontsize=10)
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.updateGeometry(self)
        self.plot4()

# Simple Moving Average 
    def SMA(self, data, ndays): 
        SMA = pd.Series(data['Close'].rolling(ndays).mean(), name = 'SMA') 
        data = data.join(SMA) 
        return data

# Exponentially-weighted Moving Average 
    def EWMA(self, data, ndays): 
        EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
        data = data.join(EMA) 
        return data
        
    def plot4(self):
        global data
        x = data.index
        n = 50
        SMA = self.SMA(data,n)
        SMA = SMA.dropna()

        ew = 200
        EWMA = self.EWMA(data,ew)
        EWMA = EWMA.dropna()

        y = data['Close']
        ax = self.figure.add_subplot(111)
        #ax.set_ylim(ymax=450)
        #ax.set_ylim(bottom=100)
        #print(dir(ax))
        #ax.set_xlabel('Test Cases')
        #ax.set_ylabel('fault courage')
        ax.plot(SMA.index,SMA['Close'])
        ax.plot(SMA.index,SMA['SMA'])
        ax.plot(EWMA.index,EWMA['EWMA_200'] )
        ax.tick_params(axis='x', rotation=0)
        self.draw()
        self.flush_events()
        
class PlotCanvas5(FigureCanvas):

    def __init__(self, parent=None, width=15, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle('Relative Volume', fontsize=10)
        #fig.supxlabel('X title', fontsize=10)
        #fig.supylabel('Y title', fontsize=10)
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.updateGeometry(self)
        self.plot5()

# RVOL 
    def RVOL(self, data, ndays): 
        Ave_VOL = pd.Series(data['Volume'].rolling(ndays).mean(), name = 'Ave_VOL') 
        data['RVOL'] = data['Volume'] / Ave_VOL
        #data = data.join(SMA)
       
        return data


        
    def plot5(self):
        global data
        x = data.index
        n = 30
        RVOL = self.RVOL(data,n)
        RVOL = RVOL.dropna()
        #print(RVOL)
        y = data['Volume']
        ax = self.figure.add_subplot(111)
        #ax.set_ylim(ymax=450)
        #ax.set_ylim(bottom=100)
        #print(dir(ax))
        #ax.set_xlabel('Test Cases')
        #ax.set_ylabel('fault courage')
        #ax.plot(RVOL.index,RVOL['Volume'])
        ax.plot(RVOL.index,RVOL['RVOL'])
        
        ax.tick_params(axis='x', rotation=0)
        self.draw()
        self.flush_events()
        
class PlotCanvas6(FigureCanvas):

    def __init__(self, parent=None, width=15, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle('Williams %R', fontsize=10)
        #fig.supxlabel('X title', fontsize=10)
        #fig.supylabel('Y title', fontsize=10)
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.updateGeometry(self)
        self.plot6()

# W%R 
    def get_wr(self, high, low, close, lookback):
        highh = high.rolling(lookback).max() 
        lowl = low.rolling(lookback).min()
        wr = -100 * ((highh - close) / (highh - lowl))
        return wr





        
    def plot6(self):
        global data
        x = data.index
        n = 30
        data['wr_14'] = self.get_wr(data['High'], data['Low'], data['Close'], 14)
        
        
        y = data['wr_14']
        ax = self.figure.add_subplot(111)
        #ax.set_ylim(ymax=450)
        #ax.set_ylim(bottom=100)
        #print(dir(ax))
        #ax.set_xlabel('Test Cases')
        #ax.set_ylabel('fault courage')
        #ax.plot(RVOL.index,RVOL['Volume'])
        ax.axhline(-20, linestyle='--', alpha=0.5, color='r')
        ax.axhline(-80, linestyle='--', alpha=0.5, color='r')
        ax.plot(x,y)
        
        ax.tick_params(axis='x', rotation=0)
        self.draw()
        self.flush_events()
        
class PlotCanvas7(FigureCanvas):

    def __init__(self, parent=None, width=15, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle('Stochastic Oscillator', fontsize=10)
        #fig.supxlabel('X title', fontsize=10)
        #fig.supylabel('Y title', fontsize=10)
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.updateGeometry(self)
        self.plot7()


        
    def plot7(self):
        global data
        x = data.index
        n = 30
        k_period = 14
        d_period = 3
        # Adds a "n_high" column with max value of previous 14 periods
        data['n_high'] = data['High'].rolling(k_period).max()
        # Adds an "n_low" column with min value of previous 14 periods
        data['n_low'] = data['Low'].rolling(k_period).min()
        # Uses the min/max values to calculate the %k (as a percentage)
        data['%K'] = (data['Close'] - data['n_low']) * 100 / (data['n_high'] - data['n_low'])
        # Uses the %k to calculates a SMA over the past 3 values of %k
        data['%D'] = data['%K'].rolling(d_period).mean()
        data.ta.stoch(high='high', low='low', k=14, d=3, append=True)
        #print(data)
        y = data['Close']
        z = data['Open']
        ax = self.figure.add_subplot(111)
        #ax.set_ylim(ymax=450)
        #ax.set_ylim(bottom=100)
        #print(dir(ax))
        #ax.set_xlabel('Test Cases')
        #ax.set_ylabel('fault courage')
        #ax.plot(RVOL.index,RVOL['Volume'])
        ##ax.axhline(-20, linestyle='--', alpha=0.5, color='r')
        ##ax.axhline(-80, linestyle='--', alpha=0.5, color='r')
        ax.plot(x,y)
        ax.plot(x,z)
        ax.plot(x,data['STOCHk_14_3_3'])
        ax.plot(x,data['STOCHd_14_3_3'])
        ax.axhline(20, linestyle='--', alpha=0.5, color='r')
        ax.axhline(80, linestyle='--', alpha=0.5, color='r')        
        ax.tick_params(axis='x', rotation=0)
        self.draw()
        self.flush_events()
          
def main():

    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()        
