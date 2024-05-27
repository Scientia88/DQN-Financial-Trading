# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:42:29 2023

@author: 220262851
"""
import numpy as np
import matplotlib.pyplot as plt
from Loader.DataManager import AssetsDataLoader
from Loader.StatesExtraction import Extract
from DeepReinforcementLearning.DRL.DQN import Train as DQNAgent
import torch as T

device = T.device("cuda" if T.cuda.is_available() else "cpu")


# Define the number of bootstrap samples
n_bootstrap_samples = 10000

# Define the risk-free rate 
risk_free_rate = 0

# Function to calculate the Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio

# DQN Parammeters 
window_size = None
Batches = 14
ReplayMemorySize = 66
TargetUpdate = 19
n_episodes = 1
steps = 18
gamma = 0.811

# Experiment Parameters
dataset = 'FTSE'
initial_investment = 1000
transaction_cost = 0.001

DataLoader = AssetsDataLoader(dataset, split_point='2021-04-08', load_from_file=False)

# Training and Test data for DQN model
dataTrain_DQN = Extract(DataLoader.data_train, 'action_select', device, gamma, steps, Batches, window_size,
                        transaction_cost)
dataTest_DQN = Extract(DataLoader.data_test, 'action_select', device, gamma, steps, Batches, window_size,
                      transaction_cost)

Agent = DQNAgent(DataLoader, dataTrain_DQN, dataTest_DQN,
                 dataset, window_size, transaction_cost,
                 Batches=Batches, gamma=gamma, ReplayMemorySize=ReplayMemorySize,
                 TargetUpdate=TargetUpdate, steps=steps)
Agent.train(n_episodes)
file_name = None

ev_DeepQNetworkTest = Agent.test(file_name=file_name, initial_investment=initial_investment, test_type='test')
DailyValue = ev_DeepQNetworkTest.GetPortfoDailyVal()
# Calculate rate of return for each day
DailyProfit = [(DailyValue[t + 1] - DailyValue[t]) / DailyValue[t] for t in range(len(DailyValue) - 1)]


# Perform bootstrap resampling
sharpe_ratios = []
for _ in range(n_bootstrap_samples):
    sample = np.random.choice(DailyProfit, size=len(DailyProfit), replace=True)
    sharpe_ratio = calculate_sharpe_ratio(sample, risk_free_rate)
    sharpe_ratios.append(sharpe_ratio)

# Calculate the mean and standard error of Sharpe ratios
mean_sharpe_ratio = np.mean(sharpe_ratios)
std_sharpe_ratio = np.std(sharpe_ratios)

# Compute the confidence interval (e.g., 95%)
confidence_level = 0.95
lower_percentile = (1 - confidence_level) / 2
upper_percentile = 1 - lower_percentile
lower_ci = np.percentile(sharpe_ratios, lower_percentile * 100)
upper_ci = np.percentile(sharpe_ratios, upper_percentile * 100)

# Plot the Sharpe ratio distribution
plt.hist(sharpe_ratios, bins=50, density=True, alpha=0.7)
plt.axvline(mean_sharpe_ratio, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(lower_ci, color='green', linestyle='dashed', linewidth=2, label=f'{int(confidence_level*100)}% CI')
plt.axvline(upper_ci, color='green', linestyle='dashed', linewidth=2)
plt.xlabel('Sharpe Ratio')
plt.ylabel('Density')
plt.legend()
#plt.title("Sharpe Ratio Distribution with Confidence Interval")
plt.show()

# Display results
print(f"Mean Sharpe Ratio: {mean_sharpe_ratio:.4f}")
print(f"Standard Error of Sharpe Ratio: {std_sharpe_ratio:.4f}")
print(f"{int(confidence_level*100)}% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")

