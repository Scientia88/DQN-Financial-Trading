
from .Environment import Env

# Define the Extract class, which inherits from the Env class
class Extract(Env):
    def __init__(self, data, action_init, device, gamma, steps, batches, window_size=1, TradingCost=0.0):
        """
        Initializes the Extract class, which extracts features for the reinforcement learning environment.

        :param data: The data used for training/testing
        :param action_init: The column name in the data representing actions
        :param device: The device (CPU/GPU) to perform computations on
        :param gamma: The discount factor for future rewards
        :param steps: The number of steps to consider for rewards
        :param batches: The number of batches for training
        :param window_size: The size of the extraction window
        :param TradingCost: The cost incurred for each transaction
        """
        RewardStartIndex = 0

        # Call the constructor of the parent class (Env) with necessary parameters
        super().__init__(data, action_init, device, gamma, steps, batches, RewardStartIndex=RewardStartIndex, TradingCost=TradingCost)

  

        # Extract and preprocess the required features from the data
        self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
        self.state_size = 4  # State size based on the preprocessed features

        # Populate the 'states' list with the preprocessed data
        for i in range(len(self.data_preprocessed)):
            self.states.append(self.data_preprocessed[i])

       


