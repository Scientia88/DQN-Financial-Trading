from collections import namedtuple
import random

# Named tuple to represent a transition in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        :param capacity: Maximum number of transitions the buffer can hold.
        """
        self.capacity = capacity
        self.buffer = []            # List to store transitions
        self.position = 0           # Pointer to keep track of the buffer position

    def push(self, *args):
        """
        Save a transition to the buffer.
        
        :param *args: A tuple representing a transition (state, action, next_state, reward).
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Initialize buffer if not at maximum capacity
        self.buffer[self.position] = args  # Store the transition tuple in the buffer
        self.position = (self.position + 1) % self.capacity  # Update the buffer position

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        :param batch_size: Number of transitions to sample.
        :return: A list of sampled transition tuples.
        """
        return random.sample(self.buffer, batch_size)  # Randomly sample transitions

    def __len__(self):
        """
        Get the current number of transitions in the buffer.
        
        :return: Number of transitions in the buffer.
        """
        return len(self.buffer)
