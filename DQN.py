import torch.nn as nn
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size, dueling = False):
        super(DQN, self).__init__()
        self.dueling = dueling
        self.Linear_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.advantage_stream = nn.Linear(64, output_size)
        self.value_stream = nn.Linear(64,1)
        
    def forward(self, x):
        x = self.Linear_layer(x)
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

            return q_values
        else:
            x = self.advantage_stream(x)
            return x
        
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return(len(self.memory))