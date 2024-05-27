import torch as T
import math as M

# Define the Env class
class Env:
    def __init__(self, data, action_init, device, gamma, steps, batches, RewardStartIndex=0, TradingCost=0):
        """
        Initializes the environment for reinforcement learning.

        :param data: The data used for training/testing
        :param action_init: The column name in the data representing actions
        :param device: The device (CPU/GPU) to perform computations on
        :param gamma: The discount factor for future rewards
        :param steps: The number of steps to consider for rewards
        :param batches: The number of batches for training
        :param RewardStartIndex: The starting index for calculating rewards
        :param TradingCost: The cost incurred for each transaction
        """
        self.data = data
        self.GetShares = False
        self.batches = batches
        self.states = []
        self.CurrentStateIndex = -1
        self.device = device
        self.steps = steps
        self.gamma = gamma
        self.action_init = action_init
        self.ActionCode = {0: 'buy', 1: 'hold', 2: 'sell'}
        self.RewardStartIndex = RewardStartIndex
        self.TradingCost = TradingCost
        self.Price = list(data.close)
        
    def step(self, action):
        """
        Takes a step in the environment based on the given action.

        :param action: The action to be taken
        :return: A tuple of (done, reward, next_state)
        """
        done = False
        next_state = None

        if (self.CurrentStateIndex + self.steps) < len(self.states):
            next_state = self.states[self.CurrentStateIndex + self.steps]
        else:
            done = True

        if action == 0:
            self.GetShares = True
        elif action == 2:
            self.GetShares = False

        reward = 0
        if not done:
            reward = self.Reward(action)

        return done, reward, next_state

    def CurrentState(self):
        """
        Gets the current state from the environment.

        :return: The current state
        """
        self.CurrentStateIndex += 1
        if self.CurrentStateIndex == len(self.states):
            return None
        return self.states[self.CurrentStateIndex]
    
    def OneStepReward(self, action, index, rewards):
        """
        Calculates reward for a single step.

        :param action: The action taken
        :param index: The index of the step
        :param rewards: List to store rewards
        """
        index += self.RewardStartIndex  
        if action == 0 or (action == 1 and self.GetShares):  
            difference = self.Price[index + 1] - self.Price[index]
            rewards.append(difference)

        elif action == 2 or (action == 1 and not self.GetShares):  
            difference = self.Price[index] - self.Price[index + 1]
            rewards.append(difference)


    def Reward(self, action):
        """
        Calculates the reward based on the given action.

        :param action: The action taken
        :return: The calculated reward
        """
        reward_index_first = self.CurrentStateIndex + self.RewardStartIndex
        reward_index_last = self.CurrentStateIndex + self.RewardStartIndex + self.steps \
            if self.CurrentStateIndex + self.steps < len(self.states) else len(self.Price) - 1

        price1 = self.Price[reward_index_first]
        price2 = self.Price[reward_index_last]

        reward = 0
        if action == 0 or (action == 1 and self.GetShares):  
            reward = ((1 - self.TradingCost) ** 2 * ((price2 - price1) / price1)) * 100  
        elif action == 2 or (action == 1 and not self.GetShares):  
            reward = ((1 - self.TradingCost) ** 2 * ((price1 - price2 ) / price2)) * 100  

        return reward
    
    
    def TotalReward(self, action_list):
        """
        Calculates the total reward for a list of actions.

        :param action_list: List of actions
        :return: The total reward
        """
        total_reward = 0
        for a in action_list:
            if a == 0:
                self.GetShares = True
            elif a == 2:
                self.GetShares = False
            self.CurrentStateIndex += 1
            total_reward += self.Reward(a)

        return total_reward
    
    def Investment(self, action_list):
        """
        Makes investments based on a list of actions.

        :param action_list: List of actions
        """
        self.data[self.action_init] = 'None'
        i = self.RewardStartIndex + 1
        for a in action_list:
            self.data[self.action_init][i] = self.ActionCode[a]
            i += 1


    def reset(self):
        """Resets the environment state."""
        self.CurrentStateIndex = -1
        self.GetShares = False

    def __iter__(self):
        """Initializes the iterator."""
        self.index_batch = 0
        self.num_batch = M.ceil(len(self.states) / self.batches)
        return self

    def __next__(self):
        """Gets the next batch of states."""
        if self.index_batch < self.num_batch:
            batch = [T.tensor([s], dtype=T.float, device=self.device) for s in
                     self.states[self.index_batch * self.batches: (self.index_batch + 1) * self.batches]]
            self.index_batch += 1
            return T.cat(batch)

        raise StopIteration


    
