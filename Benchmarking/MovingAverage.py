import numpy as np
from PerfMonitor.PerformEval import Eval
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class MAStrategy:
    def __init__(self, data_train, data_test, short_ma, long_ma, dataset_name, transaction_cost):
        self.data_train = data_train
        self.data_test = data_test
        self.dataset_name = dataset_name
        self.transaction_cost = transaction_cost
        self.short_ma = short_ma
        self.long_ma = long_ma
    
    @staticmethod
    def MovingAvg(data, period=1, column='close'):
        return data[column].rolling(window=period).mean()
    
    def test(self, test_type='train'):
        """
        Evaluates the model's performance.
        @param test_type: 'train' or 'test'
        @return: An evaluation object to access different evaluation metrics.
        """
        self.make_investment(self.data_train)
        
        if self.data_test is not None:
            self.make_investment(self.data_test)
            return Eval(self.data_train if test_type == 'train' else self.data_test, 'action_agent', 1000, self.transaction_cost)
        else:
            return Eval(self.data_train, 'action_agent', 1000, self.transaction_cost)
    
    def make_investment(self, data):
        data['action_agent'] = np.nan
        data['SSMA'] = self.MovingAvg(data, self.short_ma)  
        data['LSMA'] = self.MovingAvg(data, self.long_ma)   
        data['Signal'] = np.where(data['SSMA'] > data['LSMA'], 1, 0)
        data['Position'] = data['Signal'].diff()
        
        # data.loc[data['Position'] == 1, 'action_agent'] = 'buy'
        # data.loc[data['Position'] == -1, 'action_agent'] = 'sell'
        # data.loc[data['Position'] == 0, 'action_agent'] = 'None'
        data.loc[data['Position'] == 1, 'action_agent'] = 'buy'
        data.loc[data['Position'] == -1, 'action_agent'] = 'sell'
        data['action_agent'].fillna(method='ffill', inplace=True)  # Fill NaNs with previous values
        
        
