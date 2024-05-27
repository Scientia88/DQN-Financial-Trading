import numpy as np

class Eval:
    def __init__(self, data, action_label, initial_investment, trading_cost_ratio=0.001):
        # Initialize the evaluation class
        # data: DataFrame containing trading data
        # action_label: Column label indicating actions (buy/sell/hold)
        # initial_investment: Initial investment amount
        # trading_cost_ratio: Transaction cost ratio
        
        if not ('action' in data.columns):
            raise Exception('action is not in data columns')
        
        self.data = data
        self.initial_investment = initial_investment
        self.action_label = action_label
        self.trading_cost_ratio = trading_cost_ratio

    def TotalReturn(self):
        # Calculate and return the total return as a percentage
        
        PortfolioValue = self.GetPortfoDailyVal()  # Get daily portfolio values
        final_PortfolioValue = PortfolioValue[-1]  # Final portfolio value
        initial_investment = self.initial_investment  # Initial investment
        
        # Calculate total return as a percentage
        TotalReturn_percent = ((final_PortfolioValue - initial_investment) / initial_investment) * 100
        
        return TotalReturn_percent
    
    def GetRoR(self):
        # Calculate and return the rate of return for each day
        
        portfolio = self.GetPortfoDailyVal()  # Get daily portfolio values
        
        # Calculate rate of return for each day
        RoR = [(portfolio[t + 1] - portfolio[t]) / portfolio[t] for t in range(len(portfolio) - 1)]
        
        return RoR
    
    def TotalDailyReturn(self):
        # Calculate and return the total daily return
        
        self.ArithmReturn()  # Calculate arithmetic return
        
        return self.data[f'TotalDailyReturn_{self.action_label}'].sum()
    
    def AvgDailyReturn(self):
        # Calculate and return the average daily return
        
        self.ArithmReturn()  # Calculate arithmetic return
        return self.data[f'TotalDailyReturn_{self.action_label}'].mean()
    
    def DailyReturnVar(self, daily_return_type="arithmetic"):
        # Calculate and return the daily return variance
        
        if daily_return_type == 'arithmetic':
            self.ArithmReturn()  # Calculate arithmetic return
            return self.data[f'TotalDailyReturn_{self.action_label}'].var()
        elif daily_return_type == "logarithmic":
            self.LogReturn()  # Calculate logarithmic return
            return self.data[f'logarithmic_daily_return_{self.action_label}'].var()
    
    def GetPortfoDailyVal(self):
        # Calculate and return daily portfolio values
        
        PortfolioValue = [self.initial_investment]  # Initialize with initial investment
        self.ArithmReturn()  # Calculate arithmetic return
        num_shares = 0  # Number of shares
        prev_close = self.data.iloc[0]['close']  # Previous closing price
        
        for i in range(len(self.data)):
            action = self.data[self.action_label][i]  # Get action
            
            if action == 'buy' and num_shares == 0:
                # Buy and pay transaction cost
                num_shares = PortfolioValue[-1] * (1 - self.trading_cost_ratio) / self.data.iloc[i]['close']
                if i + 1 < len(self.data):
                    PortfolioValue.append(num_shares * self.data.iloc[i + 1]['close'])
            
            elif action == 'sell' and num_shares > 0:
                # Sell and pay transaction cost
                PortfolioValue.append(num_shares * self.data.iloc[i]['close'] * (1 - self.trading_cost_ratio))
                num_shares = 0
            
            elif (action == 'hold' or action == 'buy') and num_shares > 0:
                # Hold shares and get profit
                profit = (self.data.iloc[i]['close'] - prev_close) * num_shares
                PortfolioValue.append(PortfolioValue[-1] + profit)
            
            elif (action == 'sell' or action == 'hold') and num_shares == 0:
                # If no shares, maintain portfolio value
                PortfolioValue.append(PortfolioValue[-1])  
            
            prev_close = self.data.iloc[i]['close']  # Update previous closing price
        
        return PortfolioValue
    
    def LogReturn(self):
        # Calculate and return the logarithmic return
        
        PortfoVal = self.GetPortfoDailyVal()  # Get daily portfolio values
        LogReturn = [np.log(PortfoVal[p + 1]  / PortfoVal[p]) for p in range(len(PortfoVal) - 1)]
        TotalLogReturn = sum(LogReturn) * 100
        return TotalLogReturn
        
    def CompReturn(self):
        # Calculate and return the compounded return
        
        RawReturn = self.GetRoR()
        decimal_multipliers = np.array(RawReturn) + 1
        cumulative_multiplier = np.prod(decimal_multipliers)
        CompReturn = (cumulative_multiplier - 1) * 100
        return CompReturn
    
    def ArithmReturn(self):
        # Calculate arithmetic return and store in the data DataFrame
        
        self.data[f'TotalDailyReturn_{self.action_label}'] = 0.0  # Initialize
        
        own_share = False
        for i in range(len(self.data)):
            if (self.data[self.action_label][i] == 'buy') or (own_share and self.data[self.action_label][i] == 'hold'):
                own_share = True
                if i < len(self.data) - 1:
                    self.data[f'TotalDailyReturn_{self.action_label}'][i] = (self.data['close'][i + 1] -
                                                                             self.data['close'][i]) / \
                                                                            self.data['close'][i]
            elif self.data[self.action_label][i] == 'sell':
                own_share = False
        
        self.data[f'TotalDailyReturn_{self.action_label}'] = self.data[f'TotalDailyReturn_{self.action_label}'] * 100
    
    def SharpRatio(self):
        # Calculate and return the Sharpe ratio
        
        RoR = self.GetRoR()  # Get rate of return
        return np.mean(RoR) / np.std(RoR)




   
    


   

   
    
    
    
    
    
    
    
    
    
    
    
   