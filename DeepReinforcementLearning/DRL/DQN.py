
# DL Libraries
import torch as T
import torch.optim as optim
import torch.nn as nn

# Developed Parts
from DeepReinforcementLearning.Training import MainTrain

# Set the device to use GPU if available, otherwise use CPU
device = T.device("cuda" if T.cuda.is_available() else "cpu")

# Define a neural network class for the policy
class Network(nn.Module):
    def __init__(self, state_length, action_length):
        """
        Initializes the neural network class for the policy.

        :param state_length: The size of the input state
        :param action_length: The number of possible actions
        """
        super(Network, self).__init__()
        
        # Define the policy network layers
        self.policy_network = nn.Sequential(
            nn.Linear(state_length, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, action_length))

    def forward(self, x):
        """
        Defines the forward pass through the neural network.

        :param x: Input data
        :return: Output from the policy network
        """
        return self.policy_network(x)

# Define the training class, which inherits from MainTrain
class Train(MainTrain):
    def __init__(self, loader, Traindata, Testdata, dataset, window_size=1, transaction_cost=0.0,
                 Batches=30, gamma=0.7, ReplayMemorySize=34, TargetUpdate=5, steps=10,episodes=50):
        """
        Initializes the training class for the DeepRL agent.

        :param loader: The data loader
        :param Traindata: Training data
        :param Testdata: Test data
        :param dataset: The dataset used
        :param window_size: The size of the window for data processing
        :param transaction_cost: The cost of transactions
        :param Batches: Number of batches for training
        :param gamma: The discount factor for rewards
        :param ReplayMemorySize: Size of the replay memory
        :param TargetUpdate: Frequency of updating the target network
        :param steps: Number of steps for reward calculation
        """
        super(Train, self).__init__(loader, Traindata, Testdata, dataset, 'DeepRL', window_size,
                                    transaction_cost, Batches, gamma, ReplayMemorySize, TargetUpdate, steps, episodes)

        # Create policy and target networks, move them to device
        self.policy_net = Network(Traindata.state_size, 3).to(device)
        self.target_net = Network(Traindata.state_size, 3).to(device)
        
        # Load the policy network's initial weights into the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Define the optimizer for the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters())

        # Create a separate network for testing
        self.test_net = Network(self.Traindata.state_size, 3)
        self.test_net.to(device)
        

