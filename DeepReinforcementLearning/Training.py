# Processing Libraries
from itertools import count
from tqdm import tqdm
import math as M 
import random
import os
import inspect
from pathlib import Path

# DL Libraries
import torch as T
import torch.nn.functional as F

# Developed Parts
from PerfMonitor.PerformEval import Eval
from DeepReinforcementLearning.ReplayBuffer import ReplayBuffer, Transition

# Check if GPU is available, otherwise use CPU
device = T.device("cuda" if T.cuda.is_available() else "cpu")

# Define the main training class
class MainTrain:
    def __init__(self, loader, Traindata, Testdata, dataset, model, window_size=1, 
                 transaction_cost=0.0, Batches=30, gamma=0.7, ReplayBufferSize=50, 
                 TargetUpdate=5, steps=10, episodes=50):
        """
        Initializes the main training class for the DeepRL agent.

        :param loader: The data loader
        :param Traindata: Training data
        :param Testdata: Test data
        :param dataset: The dataset used
        :param model: The name of the model
        :param window_size: The size of the window for data processing
        :param transaction_cost: The cost of transactions
        :param Batches: Number of batches for training
        :param gamma: The discount factor for rewards
        :param ReplayBufferSize: Size of the replay memory
        :param TargetUpdate: Frequency of updating the target network
        :param steps: Number of steps for reward calculation
        """
        
        # Store dataset and configuration parameters
        self.dataset = dataset
        self.Traindata = Traindata
        self.Testdata = Testdata
        self.split_point = loader.split_point
        self.begin_date = loader.begin_date
        self.end_date = loader.end_date
        self.gamma = gamma
        self.ReplayBufferSize = ReplayBufferSize
        self.model = model
        self.window_size = window_size
        self.Batches = Batches
        self.TargetUpdate = TargetUpdate
        self.steps = steps
        self.transaction_cost = transaction_cost
        #self.episodes = episodes=50
        
        # Initialize epsilon-greedy exploration parameters
        self.EpsStart = 0.9
        self.EpsEnd = 0.05
        self.EpsDecay = 500
        self.steps_done = 0
        
        # Define the path to save results and models
        self.path_components = ['Results', self.dataset, f'{self.model}'] 
        self.current_file = inspect.getframeinfo(inspect.currentframe()).filename                
        self.PATH = os.path.join(Path(os.path.abspath(self.current_file)).parent, *self.path_components)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(self.PATH):
            os.makedirs(self.PATH)
        
        # Define the model directory
        self.model_dir = os.path.join(self.PATH, f'model.pkl')
        
        # Initialize the replay memory
        self.memory = ReplayBuffer(ReplayBufferSize)

    def save_model(self, model):
        """
        Save the model's parameters to a file.

        :param model: The model to be saved
        """
        T.save(model, self.model_dir)
        
    def Optimiser(self):
        """
        Perform optimization of the policy network.

        This function is used in the training process.
        """
        # Ensure there is enough data in the replay memory to proceed
        if len(self.memory) < self.Batches:
            return
        transitions = self.memory.sample(self.Batches)
        batch = Transition(*zip(*transitions))
        
        # Mask to track final states
        NotFinal = T.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=T.bool)
        non_final_NextStates = T.cat([s for s in batch.next_state if s is not None])
        
        # Extract batch data
        BatchState = T.cat(batch.state)
        BatchAction = T.cat(batch.action)
        BatchReward = T.cat(batch.reward)
        
        # Calculate predicted Q values and target Q values
        ST_Val = self.policy_net(BatchState).gather(1, BatchAction)
        SprimVal = T.zeros(self.Batches, device=device)
        SprimVal[NotFinal] = self.target_net(non_final_NextStates).max(1)[0].detach()
        expected_ST_Val = (SprimVal * (self.gamma ** self.steps)) + BatchReward
        
        # Compute the loss using Huber loss function
        loss = F.smooth_l1_loss(ST_Val, expected_ST_Val.unsqueeze(1))

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        for P in self.policy_net.parameters():
            P.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
        

    def ActionSelection(self, state):
        """
        Select an action based on epsilon-greedy exploration.

        :param state: The current state
        :return: The selected action
        """
        sample = random.random()
        Thresh = self.EpsEnd + (self.EpsStart - self.EpsEnd) * M.exp(-1. * self.steps_done / self.EpsDecay)
        self.steps_done += 1

        if sample > Thresh:
            # Exploitation: Choose the action with the highest Q-value
            with T.no_grad():
                self.policy_net.eval()
                action = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                return action
        else:
            # Exploration: Choose a random action
            return T.tensor([[random.randrange(3)]], device=device, dtype=T.long)

    def train(self,episodes):
        """
        Train the DeepRL agent.

        :param episodes: Number of episodes for training
        """
        print('Training is in progress', self.model, '...')
        for epis_i in tqdm(range(episodes)):
            # Reset the environment for a new episode
            self.Traindata.reset()
            state = T.tensor([self.Traindata.CurrentState()], dtype=T.float, device=device)
            for i in count():
                # Select an action
                action = self.ActionSelection(state)
                # Perform the action and get the reward and next state
                done, reward, NextState = self.Traindata.step(action.item())
                reward = T.tensor([reward], dtype=T.float, device=device)

                if NextState is not None:
                    NextState = T.tensor([NextState], dtype=T.float, device=device)

                # Store the experience in the replay memory
                self.memory.push(state, action, NextState, reward)

                if not done:
                    state = T.tensor([self.Traindata.CurrentState()], dtype=T.float, device=device)

                # Perform optimization step
                self.Optimiser()
                if done:
                    break

            if epis_i % self.TargetUpdate == 0:
                # Update the target network periodically
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Save the trained model
        self.save_model(self.policy_net.state_dict())
        print('Training is complete')

    def test(self, file_name, initial_investment=1000, test_type='test'):
        """
        Test the trained model.

        :param file_name: Name of the model file to load
        :param initial_investment: Initial investment amount
        :param test_type: Type of testing ('train' or 'test')
        :return: Action evaluation results
        """
        data = self.Traindata if test_type == 'train' else self.Testdata
        
        if file_name is None:
            self.file_name = self.model_dir
        else:
            self.file_name = file_name

        # Load the trained model
        self.test_net.load_state_dict(T.load(self.file_name))              
        self.test_net.to(device)

        # Perform testing
        List = []
        data.__iter__()
        for batch in data:
            try:
                BatchAction = self.test_net(batch).max(1)[1]
                List += list(BatchAction.cpu().numpy())
            except ValueError:
                List += [1]

        # Update investment actions in the data
        data.Investment(List)
        
        # Evaluate investment actions
        ActionEval = Eval(data.data, data.action_init, initial_investment, self.transaction_cost)
        return ActionEval
    
