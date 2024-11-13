import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math

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
    
# https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
class Noisy_DQN(nn.Module):
    def __init__(self, input, output, std_init=0.5):
        super(Noisy_DQN, self).__init__()
        self.input = input
        self.output = output
        self.std_init = std_init

        # initalize weights and biases for the linear layer
        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output, self.input))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output, self.input))

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output))

        self.register_buffer('weight_epsilon', torch.FloatTensor(self.output, self.input))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.output))

        self.reset_parameter()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu * self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu * self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)
    
    def reset_parameter(self):
        mu_range = 1/ math.sqrt(self.input)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init/math.sqrt(self.input))
        self.bias_sigma.data.fill_(self.std_init/math.sqrt(self.output))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input)
        epsilon_out = self._scale_noise(self.output)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abso().sqrt())
        return x

