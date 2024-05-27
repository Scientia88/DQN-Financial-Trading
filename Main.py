
"""
This is the main code which run DQN Algo



"""

#Visualisation Libraries
import matplotlib.pyplot as plt

#Developed parts
from Loader.DataManager import AssetsDataLoader
from Loader.StatesExtraction import Extract
from Benchmarking.MovingAverage import MAStrategy as MAS
from DeepReinforcementLearning.DRL.DQN import Train as DQNAgent
from PerfMonitor.PerformEval import Eval

# Deep Learning packages
import torch as T

device = T.device("cuda" if T.cuda.is_available() else "cpu")

def AddingTrainPortfo(model_name, portfo):
    counter = 0
    key = f'{model_name}'
    while key in train_portfolios.keys():
        counter += 1
        key = f'{model_name}{counter}'
        
    train_portfolios[key] = portfo

def AddingTestPortfo(model_name, portfo):
    counter = 0
    key = f'{model_name}'
    while key in test_portfolios.keys():
        counter += 1
        key = f'{model_name}{counter}'
    
    test_portfolios[key] = portfo

train_portfolios = {}
test_portfolios = {}



# DQN Parammeters 
window_size = None
Batches = 14
ReplayMemorySize= 66
TargetUpdate= 19
n_episodes = 1
steps = 18
gamma = 0.811


# Experiment Parameters
'''
Asset 1: 'BTC-USD', split_point='2022-01-01'
Asset 2: 'AAPL',    split_point='2021-04-07'
Asset 3: 'FTSE',    split_point='2021-04-08'

'''
dataset = 'FTSE'
initial_investment = 1000
transaction_cost = 0.001




DataLoader = AssetsDataLoader(dataset,split_point= '2021-04-08', load_from_file=False)
DataLoader.plot_data()




# Moving Average (5,20) Strategy Experiment
MovingAverageStrategy_5_20 = MAS(DataLoader.data_train, DataLoader.data_test, 5, 20, dataset,transaction_cost)   
                  

EvalMovingAverageStrategy_5_20 = MovingAverageStrategy_5_20.test(test_type= 'train')
print('MAS_5_20 train')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_5_20.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_5_20.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_5_20.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_5_20.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_5_20.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_5_20.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_5_20.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_5_20.SharpRatio()}')
MovingAverageStrategy_portfolio_train_5_20 = EvalMovingAverageStrategy_5_20.GetPortfoDailyVal()



EvalMovingAverageStrategy_5_20 = MovingAverageStrategy_5_20.test( test_type= 'test')
print('MAS_5_20 test')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_5_20.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_5_20.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_5_20.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_5_20.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_5_20.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_5_20.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_5_20.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_5_20.SharpRatio()}')
MovingAverageStrategy_portfolio_test_5_20 = EvalMovingAverageStrategy_5_20.GetPortfoDailyVal()

ModelName = f'MAS_5_20'

AddingTrainPortfo(ModelName, MovingAverageStrategy_portfolio_train_5_20)
AddingTestPortfo(ModelName, MovingAverageStrategy_portfolio_test_5_20)

# Moving Average (20,50) Strategy Experiment
MovingAverageStrategy_20_50 = MAS(DataLoader.data_train, DataLoader.data_test, 20, 50, dataset,transaction_cost)   
                  

EvalMovingAverageStrategy_20_50 = MovingAverageStrategy_20_50.test(test_type= 'train')
print('MAS_20_50 train')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_20_50.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_20_50.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_20_50.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_20_50.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_20_50.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_20_50.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_20_50.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_20_50.SharpRatio()}')
MovingAverageStrategy_portfolio_train_20_50 = EvalMovingAverageStrategy_20_50.GetPortfoDailyVal()



EvalMovingAverageStrategy_20_50 = MovingAverageStrategy_20_50.test( test_type= 'test')
print('MAS_20_50 test')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_20_50.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_20_50.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_20_50.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_20_50.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_20_50.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_20_50.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_20_50.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_20_50.SharpRatio()}')
MovingAverageStrategy_portfolio_test_20_50 = EvalMovingAverageStrategy_20_50.GetPortfoDailyVal()

ModelName = f'MAS_20_50'

AddingTrainPortfo(ModelName, MovingAverageStrategy_portfolio_train_20_50)
AddingTestPortfo(ModelName, MovingAverageStrategy_portfolio_test_20_50)

# Moving Average (50,100) Strategy Experiment
MovingAverageStrategy_50_100 = MAS(DataLoader.data_train, DataLoader.data_test, 50, 100, dataset,transaction_cost)   
                  

EvalMovingAverageStrategy_50_100 = MovingAverageStrategy_50_100.test(test_type= 'train')
print('MAS_50_100 train')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_50_100.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_50_100.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_50_100.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_50_100.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_50_100.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_50_100.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_50_100.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_50_100.SharpRatio()}')
MovingAverageStrategy_portfolio_train_50_100 = EvalMovingAverageStrategy_50_100.GetPortfoDailyVal()



EvalMovingAverageStrategy_50_100 = MovingAverageStrategy_50_100.test( test_type= 'test')
print('MAS_50_100 test')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_50_100.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_50_100.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_50_100.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_50_100.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_50_100.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_50_100.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_50_100.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_50_100.SharpRatio()}')
MovingAverageStrategy_portfolio_test_50_100 = EvalMovingAverageStrategy_50_100.GetPortfoDailyVal()

ModelName = f'MAS_50_100'

AddingTrainPortfo(ModelName, MovingAverageStrategy_portfolio_train_50_100)
AddingTestPortfo(ModelName, MovingAverageStrategy_portfolio_test_50_100)

# Moving Average (50,200) Strategy Experiment
MovingAverageStrategy_50_200 = MAS(DataLoader.data_train, DataLoader.data_test, 50, 200, dataset,transaction_cost)   
                  

EvalMovingAverageStrategy_50_200 = MovingAverageStrategy_50_200.test(test_type= 'train')
print('MAS_50_200 train')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_50_200.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_50_200.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_50_200.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_50_200.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_50_200.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_50_200.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_50_200.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_50_200.SharpRatio()}')
MovingAverageStrategy_portfolio_train_50_200 = EvalMovingAverageStrategy_50_200.GetPortfoDailyVal()



EvalMovingAverageStrategy_50_200 = MovingAverageStrategy_50_200.test( test_type= 'test')
print('MAS_50_200 test')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_50_200.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_50_200.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_50_200.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_50_200.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_50_200.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_50_200.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_50_200.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_50_200.SharpRatio()}')
MovingAverageStrategy_portfolio_test_50_200 = EvalMovingAverageStrategy_50_200.GetPortfoDailyVal()

ModelName = f'MAS_50_200'

AddingTrainPortfo(ModelName, MovingAverageStrategy_portfolio_train_50_200)
AddingTestPortfo(ModelName, MovingAverageStrategy_portfolio_test_50_200)


# Moving Average (100,200) Strategy Experiment
MovingAverageStrategy_100_200 = MAS(DataLoader.data_train, DataLoader.data_test, 100, 200, dataset,transaction_cost)   
                  

EvalMovingAverageStrategy_100_200 = MovingAverageStrategy_100_200.test(test_type= 'train')
print('MAS_100_200 train')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_100_200.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_100_200.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_100_200.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_100_200.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_100_200.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_100_200.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_100_200.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_100_200.SharpRatio()}')
MovingAverageStrategy_portfolio_train_100_200 = EvalMovingAverageStrategy_100_200.GetPortfoDailyVal()



EvalMovingAverageStrategy_100_200 = MovingAverageStrategy_100_200.test( test_type= 'test')
print('MAS_100_200 test')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {EvalMovingAverageStrategy_100_200.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {EvalMovingAverageStrategy_100_200.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {EvalMovingAverageStrategy_100_200.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {EvalMovingAverageStrategy_100_200.LogReturn()} %')
print(f'Compounded Return: --------------------------- {EvalMovingAverageStrategy_100_200.CompReturn()} %')
print(f'Average daily return: ------------------------ {EvalMovingAverageStrategy_100_200.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {EvalMovingAverageStrategy_100_200.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {EvalMovingAverageStrategy_100_200.SharpRatio()}')
MovingAverageStrategy_portfolio_test_100_200 = EvalMovingAverageStrategy_100_200.GetPortfoDailyVal()

ModelName = f'MAS_100_200'

AddingTrainPortfo(ModelName, MovingAverageStrategy_portfolio_train_100_200)
AddingTestPortfo(ModelName, MovingAverageStrategy_portfolio_test_100_200)

'''
This section dedicates to the Deep Q-Learning agent without encoder part. 
TargetUpdate is how frequent the policy network is hard-copied to the target network in terms of number 
of episodes. n_actions is set to 3 because we have buy, sell and none actions. n_episodes is the number 
of episodes to run the algorithm. EPS is the  𝜖 in the  𝜖-greedy method.

'''



# Training and Test data for DQN model
dataTrain_DQN = Extract(DataLoader.data_train,'action_select', device, gamma, steps, Batches, window_size, transaction_cost)
dataTest_DQN = Extract(DataLoader.data_test,'action_select', device, gamma, steps, Batches, window_size, transaction_cost)


Agent = DQNAgent(DataLoader, dataTrain_DQN, dataTest_DQN, 
                     dataset, window_size, transaction_cost,
                     Batches=Batches, gamma=gamma, ReplayMemorySize=ReplayMemorySize,
                     TargetUpdate=TargetUpdate, steps=steps)

Agent.train(n_episodes)
file_name = None

ev_DeepQNetworkTrain = Agent.test(file_name=file_name,initial_investment=initial_investment, test_type='train')
DeepQNetwork_portfolio_train = ev_DeepQNetworkTrain.GetPortfoDailyVal()

print('DQN train')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {ev_DeepQNetworkTrain.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {ev_DeepQNetworkTrain.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {ev_DeepQNetworkTrain.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {ev_DeepQNetworkTrain.LogReturn()} %')
print(f'Compounded Return: --------------------------- {ev_DeepQNetworkTrain.CompReturn()} %')
print(f'Average daily return: ------------------------ {ev_DeepQNetworkTrain.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {ev_DeepQNetworkTrain.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {ev_DeepQNetworkTrain.SharpRatio()}')

ev_DeepQNetworkTest = Agent.test(file_name=file_name,initial_investment=initial_investment, test_type='test')
DeepQNetwork_portfolio_test = ev_DeepQNetworkTest.GetPortfoDailyVal()

print('DQN test')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {ev_DeepQNetworkTest.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {ev_DeepQNetworkTest.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {ev_DeepQNetworkTest.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {ev_DeepQNetworkTest.LogReturn()} %')
print(f'Compounded Return: --------------------------- {ev_DeepQNetworkTest.CompReturn()} %')
print(f'Average daily return: ------------------------ {ev_DeepQNetworkTest.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {ev_DeepQNetworkTest.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {ev_DeepQNetworkTest.SharpRatio()}')
ModelName = 'DQN'

AddingTrainPortfo(ModelName, DeepQNetwork_portfolio_train)
AddingTestPortfo(ModelName, DeepQNetwork_portfolio_test)

#Buy the stock in the beginning of the process and hold it until the end without selling.

DataLoader.data_train['action_deepRL'] = 'buy'
ev_BandH = Eval(DataLoader.data_train, 'action_deepRL', initial_investment)
print('B snd H train')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {ev_BandH.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {ev_BandH.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {ev_BandH.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {ev_BandH.LogReturn()} %')
print(f'Compounded Return: --------------------------- {ev_BandH.CompReturn()} %')
print(f'Average daily return: ------------------------ {ev_BandH.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {ev_BandH.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {ev_BandH.SharpRatio()}')
BandH_portfolio_train = ev_BandH.GetPortfoDailyVal()

DataLoader.data_test['action_deepRL'] = 'buy'
ev_BandH = Eval(DataLoader.data_test, 'action_deepRL', initial_investment)
print('B snd H test')
print(f'Initial Investment: -------------------------- {initial_investment} $')
print(f'Final Portfolio Value: ----------------------- {ev_BandH.GetPortfoDailyVal()[-1]} $')
print(f'Total Return: -------------------------------- {ev_BandH.TotalReturn()} %')
print(f'Arithmetic Return: --------------------------- {ev_BandH.TotalDailyReturn()} %')
print(f'Logarithmic Return: -------------------------- {ev_BandH.LogReturn()} %')
print(f'Compounded Return: --------------------------- {ev_BandH.CompReturn()} %')
print(f'Average daily return: ------------------------ {ev_BandH.AvgDailyReturn()} %')
print(f'Daily return variance: ----------------------- {ev_BandH.DailyReturnVar()}')
print(f'Sharpe Ratio: --------------------------------- {ev_BandH.SharpRatio()}')
BandH_portfolio_test = ev_BandH.GetPortfoDailyVal()

AddingTrainPortfo('B&H', BandH_portfolio_train)
AddingTestPortfo('B&H', BandH_portfolio_test)

# Visualisation of training experiments

# Create subplots
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 15))

# Plot portfolio values for each experiment in subplots
for i, (experiment_name, portfolio_values) in enumerate(train_portfolios.items()):
    ax = axes[i]
    ax.plot(portfolio_values, label=experiment_name)
    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio Value")
    ax.set_title(f'{experiment_name} in training period')
    ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Visualisation of testing experiments

# Create subplots
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 15))

# Plot portfolio values for each experiment in subplots
for i, (experiment_name, portfolio_values) in enumerate(test_portfolios.items()):
    ax = axes[i]
    ax.plot(portfolio_values, label=experiment_name)
    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio Value")
    ax.set_title(f'{experiment_name} in test period')
    ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# Visualisation of training experiments

# Create a single subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot portfolio values for each experiment in the same subplot
for experiment_name, portfolio_values in train_portfolios.items():
    ax.plot(portfolio_values, label=experiment_name)

ax.set_xlabel("Day")
ax.set_ylabel("Portfolio Value")
#ax.set_title("Training Experiments")
ax.legend()

# Show the plot
plt.show()




# Visualisation of testing experiments

# Create a single subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot portfolio values for each experiment in the same subplot
for experiment_name, portfolio_values in test_portfolios.items():
    ax.plot(portfolio_values, label=experiment_name)

ax.set_xlabel("Day")
ax.set_ylabel("Portfolio Value")
#ax.set_title("Testing Experiments")
ax.legend()

# Show the plot
plt.show()



