import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math

class DQN(nn.Module):
    def __init__(self, input_size, max_notes, action_space, dueling = False, std_init=0.3, noisy=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        # conv1d because it need to caputre the velocity from stacked note frame
        self.conv_layer = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.linear_layer = nn.Sequential(
            Noisy_Layer(64*3*max_notes, 128, std_init) if noisy 
            else nn.Linear(64*3*max_notes, 128),
            nn.ReLU(inplace=True)
        )
        self.value_stream = Noisy_Layer(128, 1, std_init) if noisy else nn.Linear(128,1)
        # split the actions into 4 layers that each represents the key action
        if noisy:
            self.advantage_stream = nn.ModuleList([
                Noisy_Layer(128, dim, std_init) for dim in action_space
            ])
        else:
            self.advantage_stream = nn.ModuleList([
                nn.Linear(128, dim) for dim in action_space
            ])
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        if self.dueling:
            value = self.value_stream(x).unsqueeze(1)
            advantage = [adv(x).unsqueeze(1) for adv in self.advantage_stream]
            advantage = torch.cat(advantage, dim=1)

            q_values = value + (advantage - advantage.mean(dim=2, keepdim=True))

            return q_values
        else:
            # unsqueeze to separate q values per key action
            # return shape (batch, 1, action) per keys
            q_values = [adv(x).unsqueeze(1) for adv in self.advantage_stream]

            # concat to get shape (batch, key, action)
            q_values = torch.cat(q_values, dim=1)
            
            return q_values
        
class LSTM_DQN(nn.Module):
    def __init__(self, input_size, action_space, hidden_size=128, num_layers=1, dueling = False, std_init=0.3, noisy=False, dropout=0.2):
        super(LSTM_DQN, self).__init__()
        self.dueling = dueling
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.linear_layer = nn.Sequential(
             nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True)
        )
        self.value_stream = Noisy_Layer(128, 1, std_init) if noisy else nn.Linear(128,1)
        # split the actions into 4 layers that each represents the key action
        if noisy:
            self.advantage_stream = nn.ModuleList([
                Noisy_Layer(128, dim, std_init) for dim in action_space
            ])
        else:
            self.advantage_stream = nn.ModuleList([
                nn.Linear(128, dim) for dim in action_space
            ])
    
    def forward(self, x, hidden_state=None):
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out[:, -1, :] # shape (batch_size, hidden_size)

        lstm_out = self.linear_layer(lstm_out)
        if self.dueling:
            value = self.value_stream(lstm_out).unsqueeze(1)
            advantage = [adv(lstm_out).unsqueeze(1) for adv in self.advantage_stream]
            advantage = torch.cat(advantage, dim=1)

            q_values = value + (advantage - advantage.mean(dim=2, keepdim=True))

            return q_values, hidden_state
        else:
            # unsqueeze to separate q values per key action
            # return shape (batch, 1, action) per keys
            q_values = [adv(lstm_out).unsqueeze(1) for adv in self.advantage_stream]

            # concat to get shape (batch, key, action)
            q_values = torch.cat(q_values, dim=1)
            
            return q_values, hidden_state
        
class LSTM_PPO(nn.Module):
    def __init__(self, input_size, action_space, hidden_size=128, num_layers=1, dropout=0):
        super(LSTM_DQN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.linear_layer = nn.Sequential(
             nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True)
        )

        self.actors = nn.ModuleList([
            nn.Linear(hidden_size, dim) for dim in action_space
        ])

        self.critic = nn.Linear(128, 1)

    def forward(self, x, hidden_state=None):
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out[:, -1, :]
        logits = [actor(lstm_out).unsqueeze(1) for actor in self.actors]
        logits = torch.cat(logits, dim=1)
        value = self.critic(lstm_out)

        return logits, value, hidden_state

        
class ReplayMemory:
    def __init__(self, capacity=None, ppo = False):
        self.memory = deque(maxlen=capacity)
        self.ppo = ppo
    
    def push(self, state, action, reward, next_state, done):
        if not self.ppo:
            self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return(len(self.memory))
    
    def clear(self):
        self.memory.clear()
    
# https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
class Noisy_Layer(nn.Module):
    def __init__(self, input, output, std_init):
        super(Noisy_Layer, self).__init__()
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
            self.reset_noise()
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
        x = x.sign().mul(x.abs().sqrt())
        return x

