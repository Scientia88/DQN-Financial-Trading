# pip install optuna
# pip install plotly
# pip install --upgrade --user optuna matplotlib
import optuna
import optuna.visualization as optuna_viz
import torch as T
#Developed parts
from Loader.DataManager import AssetsDataLoader
from Loader.StatesExtraction import Extract
from DeepReinforcementLearning.DRL.DQN import Train as DQNAgent



def objective(trial):
    gamma = trial.suggest_float('gamma', 0.8, 0.99)
    TargetUpdate = trial.suggest_int('TargetUpdate', 5, 30)
    ReplayMemorySize = trial.suggest_int('ReplayMemorySize', 10, 50)
    Batches = trial.suggest_int('Batches', 10, 50)
    steps = trial.suggest_int('steps', 5, 20)
    
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    
    # DQN Parammeters 
    window_size = None
    n_episodes = 1
    
    
    # Experiment Parameters
    dataset = 'AAPL'
    initial_investment = 1000
    transaction_cost = 0.0
    
    DataLoader = AssetsDataLoader(dataset,split_point='2022-01-01', load_from_file=False)
    dataTrain_DQN = Extract(DataLoader.data_train,'action_encoder_decoder', device, gamma, steps, Batches, window_size, transaction_cost)
    dataTest_DQN = Extract(DataLoader.data_test,'action_encoder_decoder', device, gamma, steps, Batches, window_size, transaction_cost)
    
    Agent = DQNAgent(DataLoader, dataTrain_DQN, dataTest_DQN, 
                     dataset, window_size, transaction_cost,
                     Batches=Batches, gamma=gamma, ReplayMemorySize=ReplayMemorySize,
                     TargetUpdate=TargetUpdate, steps=steps)
    Agent.train(n_episodes)
    file_name = None
    
    ev_DeepQNetworkTrain = Agent.test(file_name=file_name,initial_investment=initial_investment, test_type='train')
    return  ev_DeepQNetworkTrain.TotalReturn()

study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
optuna_viz.plot_optimization_history(study)

print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

  


