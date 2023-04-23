import backtrader as bt
import pandas as pd
import numpy as np
import datetime as dt
from tensorflow import keras
from pickle import load
from keras.models import load_model
import matplotlib.pyplot as plt

Lenght = 1000

# Define the strategy class
class TradingSignalStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Load the LSTM model
        self.model = load_model('ClassificationData/Models/Classificator.h5')

        self.dataclose = self.datas[0].close
        self.X_test = np.load("ClassificationData/X_test.npy", allow_pickle=True)

        self.i = 0
        self.previousBuy = False

    def next(self):

        self.pred = self.model.predict(self.X_test[self.i].reshape(1, 7, 9))

        if self.pred[0][0] > self.pred[0][1] and self.pred[0][0] >= 0.69 and self.previousBuy != True:
            self.buy(price=self.dataclose[0])
            self.log('BUY CREATE, %.4f' % self.dataclose[0])
            print(self.i)
            self.previousBuy = True

        # If the model predicts a higher probability of selling than buying, sell
        elif self.pred[0][0] < self.pred[0][1] and self.pred[0][1] >= 0.69 and self.previousBuy != False:
            self.sell(price=self.dataclose[0])
            self.log('SELL CREATE, %.4f' % self.dataclose[0])
            print(self.i)
            self.previousBuy = False

        if self.i <= Lenght:
            self.i += 1


class PredictionStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):

        self.dataclose = self.datas[0].close

        self.model = load_model('PredictionData/Models/PredictionGRU.h5')
        self.X_test = np.load("PredictionData/X_test.npy", allow_pickle=True)
        self.y_scaler = load(open('PredictionData/y_scaler.pkl', 'rb'))

        self.i = 0
        self.previousBuy = False

    def next(self):

        self.pred = self.model.predict(self.X_test[self.i].reshape(1, 30, 9))
        self.rescaled_pred = self.y_scaler.inverse_transform(self.pred)
        self.pred_mean = self.rescaled_pred[0].mean()

        if self.pred_mean > self.dataclose[0]+(self.dataclose[0]*0.005) and self.previousBuy != True:
            self.buy(price=self.dataclose[0])
            self.log('BUY CREATE, %.4f' % self.dataclose[0])
            print(self.i)
            self.previousBuy = True

        # If the model predicts a higher probability of selling than buying, sell
        elif self.pred_mean < self.dataclose[0]-(self.dataclose[0]*0.005) and self.previousBuy != False:
            self.sell(price=self.dataclose[0])
            self.log('SELL CREATE, %.4f' % self.dataclose[0])
            print(self.i)
            self.previousBuy = False

        if self.i < data_len - 1:
            self.i += 1


# Define the function to load data
def load_classification_data():
    dataset = pd.read_csv('EURUSD_DATASET.csv', parse_dates=True)
    dataset = dataset.drop("ZigZag", axis='columns')

    dataset['Volume'] = 0
    dataset['Open Interest'] = 0

    dataset.Date = pd.to_datetime(dataset.Date, dayfirst=True)
    dataset = dataset.set_index(dataset.Date)
    dataset = dataset.drop("Date", axis='columns')

    n_test = len(dataset) - 2140
    dataset = dataset.iloc[n_test:]

    # n = len(dataset) - Lenght
    # dataset = dataset.iloc[:-n]

    # sort by date in ascending order
    dataset = dataset.sort_values(by='Date')

    print(dataset)
    print(dataset.dtypes)
    data_feed = bt.feeds.PandasData(dataname=dataset)

    return data_feed


def load_prediction_data():

    dataset = pd.read_csv('PredictionData/Simulation_Dataset.csv', parse_dates=True)
    dataset['Volume'] = 0
    dataset['Open Interest'] = 0

    dataset.Date = pd.to_datetime(dataset.Date, dayfirst=True)
    dataset = dataset.set_index(dataset.Date)
    dataset = dataset.drop("Date", axis='columns')

    n = len(dataset) - Lenght
    dataset = dataset.iloc[:-n]

    # sort by date in ascending order
    # dataset = dataset.sort_values(by='Date')

    print(dataset)
    print(dataset.dtypes)
    data_feed = bt.feeds.PandasData(dataname=dataset)

    return data_feed, len(dataset)


if __name__ == '__main__':
    # Create a cerebro instance
    cerebro = bt.Cerebro()

    ######## CLASSIFICATION #########
    # # Add the strategy
    # cerebro.addstrategy(TradingSignalStrategy)
    # # Load the data
    # data = load_classification_data()


    ######## PREDICTION #########
    # Load the data
    data, data_len = load_prediction_data()
    # Add the strategy
    cerebro.addstrategy(PredictionStrategy)

    cerebro.adddata(data)

    # Set the initial cash balance
    cerebro.broker.setcash(1000.0)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    # Set the commission and slippage
    # cerebro.broker.setcommission(commission=10)
    # cerebro.broker.set_slippage_fixed(0.01, slip_open=True)

    # Run the simulation
    cerebro.run()

    # Print the final portfolio value
    print('Final portfolio value: %.2f' % cerebro.broker.getvalue())

    plot1 = cerebro.plot(iplot=True, volume=False, width=32, height=16)
