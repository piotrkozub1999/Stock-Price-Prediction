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
SimulationYear = 2021
### Stock symbol
code = "EURUSD"
### Dataset for backtesting simulation path
SimulationDatasetPath = "SimulationDatasets/"+SimulationType+"/"+code+"_"+str(SimulationYear)+".csv"
### Creating directory for learning data
LearningDataPath = SimulationType+"Data/"+code+"_"+str(SimulationYear)

DatasetPath = 'EURUSD_2000_2023.csv'
# Define the strategy class
class TradingSignalStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Load the LSTM model
        self.model = load_model(LearningDataPath+"/Classification10Days.h5")

        self.dataclose = self.datas[0].close
        self.X_test = np.load(LearningDataPath+"/X_test.npy", allow_pickle=True)

        self.i = 0
        # self.i = 2
        self.previousBuy = False
        self.size = 0


    def next(self):

        self.pred = self.model.predict(self.X_test[self.i].reshape(1, 10, 9), verbose=0)

        if self.pred[0][0] > self.pred[0][1] and self.pred[0][0] >= 0.75 and self.previousBuy != True:
            self.size = int(self.broker.getvalue() / self.dataclose[0])
            # print("Cash: " + str(self.broker.getvalue()))
            # print("Size: " + str(self.size))
            # self.buy(size=self.size, price=self.dataclose[0])
            self.buy(price=self.dataclose[0])
            self.log('BUY CREATE, %.4f' % self.dataclose[0])
            print(self.i)
            self.previousBuy = True

        # If the model predicts a higher probability of selling than buying, sell
        elif self.pred[0][0] < self.pred[0][1] and self.pred[0][1] >= 0.60 and self.previousBuy != False:
            # print("Size: " + str(self.size))
            # self.sell(size=self.size, price=self.dataclose[0])
            self.sell(price=self.dataclose[0])
            self.log('SELL CREATE, %.4f' % self.dataclose[0])
            print(self.i)
            self.previousBuy = False

        if self.i < data_len-2:
            self.i += 1


#     def next(self):
#         if self.i < 2:
#             self.i += 1
#         else:
#             self.pred1 = self.model.predict(self.X_test[self.i-2].reshape(1, 10, 9), verbose=0)
#             self.pred2 = self.model.predict(self.X_test[self.i-1].reshape(1, 10, 9), verbose=0)
#             self.pred3 = self.model.predict(self.X_test[self.i].reshape(1, 10, 9), verbose=0)
#             # buyMean = (self.pred1[0][0] + self.pred2[0][0] + self.pred3[0][0])/3
#             # sellMean = (self.pred1[0][1] + self.pred2[0][1] + self.pred3[0][1])/3
#
#             if self.pred1[0][0] > self.pred1[0][1] and self.pred2[0][0] > self.pred2[0][1] and self.pred3[0][0] > self.pred3[0][1] and self.previousBuy != True:
#                 # self.size = int(self.broker.getvalue() / self.dataclose[0])
#                 # print("Cash: " + str(self.broker.getvalue()))
#                 # print("Size: " + str(self.size))
#                 # self.buy(size=self.size, price=self.dataclose[0])
#                 self.buy(price=self.dataclose[0])
#                 self.log('BUY CREATE, %.4f' % self.dataclose[0])
#                 print(self.i)
#                 self.previousBuy = True
#
#             # If the model predicts a higher probability of selling than buying, sell
#             elif self.pred1[0][0] < self.pred1[0][1] and self.pred2[0][0] < self.pred2[0][1] and self.pred3[0][0] < self.pred3[0][1] and self.previousBuy != False:
#                 # print("Size: " + str(self.size))
#                 # self.sell(size=self.size, price=self.dataclose[0])
#                 self.sell(price=self.dataclose[0])
#                 self.log('SELL CREATE, %.4f' % self.dataclose[0])
#                 # print("Cash: " + str(self.broker.get_cash()))
#                 print(self.i)
#                 self.previousBuy = False
#
#             if self.i < data_len-2:
#                 self.i += 1


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

        if self.pred_mean > self.dataclose[0] + (self.dataclose[0] * 0.005) and self.previousBuy != True:
        # if self.pred_mean > self.dataclose[0]+(self.dataclose[0]*0.005):
        #     self.size = int(self.broker.getvalue() / self.dataclose[0])
        #     print("Cash: " + str(self.broker.getvalue()))
        #     print("Size: " + str(self.size))
        #     self.buy(size=self.size, price=self.dataclose[0])
            self.buy(price=self.dataclose[0])
            self.log('BUY CREATE, %.4f' % self.dataclose[0])
            print(self.i)
            self.previousBuy = True

        # If the model predicts a higher probability of selling than buying, sell
        elif self.pred_mean < self.dataclose[0] - (self.dataclose[0] * 0.0001) and self.previousBuy != False:
        # elif self.pred_mean < self.dataclose[0]-(self.dataclose[0]*0.005):
        #     print("Size: " + str(self.size))
            # self.sell(size=self.size, price=self.dataclose[0])
            self.sell(price=self.dataclose[0])
            self.log('SELL CREATE, %.4f' % self.dataclose[0])
            # print("Cash: " + str(self.broker.get_cash()))
            print(self.i)
            self.previousBuy = False

        if self.i < data_len-2:
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



if __name__ == '__main__':
    # Create a cerebro instance
    cerebro = bt.Cerebro()
    # Load the data
    data, data_len = load_data()
    print(data_len)

    if SimulationType == "Classification":
        ####### CLASSIFICATION #########
        cerebro.addstrategy(TradingSignalStrategy)
    else:
        ######## PREDICTION #########
        cerebro.addstrategy(PredictionStrategy)

    cerebro.adddata(data)

    # Set the initial cash balance
    cerebro.broker.setcash(1000.0)

    # cerebro.addsizer(bt.sizers.AllInSizer)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # Run the simulation
    cerebro.run()

    # Print the final portfolio value
    print('Final portfolio value: %.2f' % cerebro.broker.getvalue())

    plot1 = cerebro.plot(iplot=True, volume=False)
