import backtrader as bt
import pandas as pd
import numpy as np
import datetime as dt
from tensorflow import keras
from pickle import load
from keras.models import load_model
import matplotlib.pyplot as plt

# SimulationType = "Classification"
SimulationType = "Prediction"

### Setting simulation year and learning period
SimulationYear = 2019
### Stock symbol
code = "EURUSD"
### Dataset for backtesting simulation path
SimulationDatasetPath = "SimulationDatasets/"+SimulationType+"/"+code+"_"+str(SimulationYear)+".csv"
### Creating directory for learning data
LearningDataPath = SimulationType+"Data/"+code+"_"+str(SimulationYear)

DatasetPath = 'EURUSD_2000_2023.csv'

Test_data = np.load(LearningDataPath+"/X_test.npy", allow_pickle=True)
model = load_model(LearningDataPath+"/PredictionModel.h5")
X_test = np.load(LearningDataPath+"/X_test.npy", allow_pickle=True)
y_scaler = load(open(LearningDataPath+"/y_scaler.pkl", "rb"))

ranges = [(0, 500)]
buy_values = []
sell_values = []

class SingleSignal(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Load the LSTM model
        self.model = load_model(LearningDataPath+"/Classification_Model.h5")
        self.dataclose = self.datas[0].close
        self.X_test = np.load(LearningDataPath+"/X_test.npy", allow_pickle=True)
        self.i = 0
        self.previousBuy = False
        self.size = 0
        self.n_steps_in = (len(self.X_test[0]))


    def next(self):

        self.pred = self.model.predict(self.X_test[self.i].reshape(1, self.n_steps_in, 9), verbose=0)
        if any(lower_bound < self.i < upper_bound for lower_bound, upper_bound in
               ranges) or self.i in buy_values or self.i in sell_values:
            if self.previousBuy != True and self.pred[0][0] > self.pred[0][1] and self.pred[0][0] >= 0.70 or self.i in buy_values:
                self.size = int(self.broker.getvalue() / self.dataclose[0])
                print(self.pred[0][0])
                self.buy(price=self.dataclose[0])
                self.log('BUY CREATE, %.4f' % self.dataclose[0])
                print(self.i)
                self.previousBuy = True

            # If the model predicts a higher probability of selling than buying, sell
            elif self.previousBuy != False and self.pred[0][0] < self.pred[0][1] and self.pred[0][
                1] >= 0.70 or self.i in sell_values:
                # print("Size: " + str(self.size))
                # self.sell(size=self.size, price=self.dataclose[0])
                print(self.pred[0][1])
                self.sell(price=self.dataclose[0])
                self.log('SELL CREATE, %.4f' % self.dataclose[0])
                print(self.i)
                self.previousBuy = False

        if self.i < data_len - 2:
            self.i += 1


class PredictionStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):

        self.dataclose = self.datas[0].close

        self.model = load_model(LearningDataPath+"/PredictionModel.h5")
        self.X_test = np.load(LearningDataPath+"/X_test.npy", allow_pickle=True)
        self.y_scaler = load(open(LearningDataPath+"/y_scaler.pkl", "rb"))

        self.i = 0
        self.previousBuy = False

    def next(self):

        self.pred = self.model.predict(self.X_test[self.i].reshape(1, 30, 9), verbose=0)
        self.rescaled_pred = self.y_scaler.inverse_transform(self.pred)
        self.pred_mean = self.rescaled_pred[0].mean()
        if any(lower_bound < self.i < upper_bound for lower_bound, upper_bound in
               ranges) or self.i in buy_values or self.i in sell_values:
            if self.pred_mean > self.dataclose[0] + (self.dataclose[0] * 0.003) and self.previousBuy != True or self.i in buy_values:
                self.buy(price=self.dataclose[0])
                self.log('BUY CREATE, %.4f' % self.dataclose[0])
                print(self.i)
                self.previousBuy = True

            # If the model predicts a higher probability of selling than buying, sell
            elif self.pred_mean < self.dataclose[0] - (self.dataclose[0] * 0.003) and self.previousBuy != False or self.i in sell_values:
                self.sell(price=self.dataclose[0])
                self.log('SELL CREATE, %.4f' % self.dataclose[0])
                print(self.i)
                self.previousBuy = False

        # if self.i <= data_len+1:
        self.i += 1

class PredictionStrategy2(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):

        self.dataclose = self.datas[0].close

        self.model = load_model(LearningDataPath+"/PredictionModel.h5")
        self.X_test = np.load(LearningDataPath+"/X_test.npy", allow_pickle=True)
        self.y_scaler = load(open(LearningDataPath+"/y_scaler.pkl", "rb"))

        self.i = 0
        self.previousBuy = False

    def next(self):

        self.pred = self.model.predict(self.X_test[self.i].reshape(1, 30, 9), verbose=0)
        self.rescaled_pred = self.y_scaler.inverse_transform(self.pred)
        self.first_mean = (self.rescaled_pred[0][0] + self.rescaled_pred[0][1]) / 2
        self.second_mean = (self.rescaled_pred[0][2] + self.rescaled_pred[0][3] + self.rescaled_pred[0][4]) / 3
        if any(lower_bound < self.i < upper_bound for lower_bound, upper_bound in
               ranges) or self.i in buy_values or self.i in sell_values:
            if self.second_mean > self.first_mean + 0.0001 and self.previousBuy != True or self.i in buy_values:
                self.buy(price=self.dataclose[0])
                self.log('BUY CREATE, %.4f' % self.dataclose[0])
                print(self.i)
                self.previousBuy = True

            # If the model predicts a higher probability of selling than buying, sell
            elif self.first_mean > self.second_mean + 0.0001 and self.previousBuy != False or self.i in sell_values:
                self.sell(price=self.dataclose[0])
                self.log('SELL CREATE, %.4f' % self.dataclose[0])
                print(self.i)
                self.previousBuy = False

        # if self.i <= data_len+1:
        self.i += 1


# Define the function to load data
def load_data():
    dataset = pd.read_csv(SimulationDatasetPath, parse_dates=True)

    dataset['Volume'] = 0
    dataset['Open Interest'] = 0

    dataset.Date = pd.to_datetime(dataset.Date, dayfirst=True)
    dataset = dataset.set_index(dataset.Date)
    dataset = dataset.drop("Date", axis='columns')

    print(dataset)
    print(dataset.dtypes)
    data_feed = bt.feeds.PandasData(dataname=dataset)

    return data_feed, len(dataset)


def load_pred_data():
    model = load_model(LearningDataPath + "/PredictionModel.h5")
    X_test = np.load(LearningDataPath + "/X_test.npy", allow_pickle=True)
    y_scaler = load(open(LearningDataPath + "/y_scaler.pkl", "rb"))
    i = 0
    pred_data = []
    while i < len(X_test):
        pred = model.predict(X_test[i].reshape(1, 30, 9), verbose=0)
        rescaled_pred = y_scaler.inverse_transform(pred)
        pred_mean = rescaled_pred[0].mean()
        pred_data.append(pred_mean)
        i += 1

    dataset = pd.read_csv(SimulationDatasetPath, parse_dates=True)
    dataset.Date = pd.to_datetime(dataset.Date, dayfirst=True)
    predDataFrame = pd.DataFrame()
    predDataFrame["Close"] = pred_data
    predDataFrame = predDataFrame.set_index(dataset.Date)
    data = bt.feeds.PandasData(dataname=predDataFrame)

    return data


if __name__ == '__main__':
    # Create a cerebro instance
    cerebro = bt.Cerebro()
    # Load the data
    data, data_len = load_data()

    pred_data = load_pred_data()

    print(data_len)

    if SimulationType == "Classification":
        ####### CLASSIFICATION #########
        cerebro.addstrategy(SingleSignal)
        # cerebro.addstrategy(ThreeSignals)
    else:
        ######## PREDICTION #########
        # cerebro.addstrategy(PredictionStrategy)
        cerebro.addstrategy(PredictionStrategy2)

    cerebro.adddata(data)


    pred_data.compensate(data)  # let the system know ops on data1 affect data0
    pred_data.plotinfo.plotmaster = data
    pred_data.plotinfo.sameaxis = True
    cerebro.adddata(pred_data)
    # Set the initial cash balance

    cerebro.broker.setcash(1000.0)

    # cerebro.addsizer(bt.sizers.AllInSizer)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # Run the simulation
    cerebro.run()

    # Print the final portfolio value
    print('Final portfolio value: %.2f' % cerebro.broker.getvalue())

    plot1 = cerebro.plot(iplot=True, volume=False)
