# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from Loader.DataManager import AssetsDataLoader
from Loader.StatesExtraction import Extract
from DeepReinforcementLearning.DRL.DQN import Train as DQNAgent
import torch as T

device = T.device("cuda" if T.cuda.is_available() else "cpu")

# DQN Parammeters 
window_size = None
Batches = 14
ReplayMemorySize = 66
TargetUpdate = 19
n_episodes = 1
steps = 18
gamma = 0.811

# Experiment Parameters
dataset = 'AAPL'
initial_investment = 1000

# List of transaction cost values to test
transaction_costs = [0.001, 0.005, 0.010, 0.015, 0.020]

# Define the number of samples per transaction cost
num_samples_per_cost = 50

# Create an empty DataFrame to store the results
results_df = pd.DataFrame() 

# Loop through the transaction cost values
for transaction_cost in transaction_costs:
    sample_values = []
    for _ in range(num_samples_per_cost):
        DataLoader = AssetsDataLoader(dataset, split_point='2021-04-07', load_from_file=False)

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
        DeepQNetwork_portfolio_test = ev_DeepQNetworkTest.GetPortfoDailyVal()[-1]

        # Append the result to the list of sample values
        sample_values.append(DeepQNetwork_portfolio_test)

    # Add the list of sample values to the DataFrame as a new column
    results_df[f'Sample_{transaction_cost}'] = sample_values

# Print or use the DataFrame as needed
print(results_df)

# To save the DataFrame to a CSV file
results_df.to_csv('D:\Indipendent Learning\Dissertation\DQN_Trading_Strategy - Final\TC_Sensitivity_Analysis.csv', index=False)

# Perform one-way ANOVA (Kruskal-Wallis H-test)
stats.kruskal(results_df['Sample_0.001'],results_df['Sample_0.005'],
              results_df['Sample_0.01'],results_df['Sample_0.015'],results_df['Sample_0.02'])


# Assuming 'results_df' is your DataFrame with five columns
data_columns = results_df.columns.values.tolist()

# Create a box plot for each column
plt.figure(figsize=(10, 6))
plt.boxplot([results_df[col] for col in data_columns], labels=data_columns)
#plt.title("Box Plot of Results by Transaction Cost")
plt.xlabel("Transaction Cost")
plt.ylabel("Final Fund's Value")
plt.show()






