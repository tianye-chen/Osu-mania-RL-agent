import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math

class DQN(nn.Module):
    def __init__(self, input_size, max_notes, action_space, dueling = False):
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
            nn.Linear(64*3*max_notes, 128),
            nn.ReLU(inplace=True)
        )

        self.value_stream = nn.Linear(128,1)

        # split the actions into 4 layers that each represents the key action
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
    def __init__(self, input_size, action_space, hidden_size=128, num_layers=1, dueling = False, dropout=0.2):
        super(LSTM_DQN, self).__init__()
        self.dueling = dueling
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True)
        )
        self.value_stream = nn.Linear(128,1)
        # split the actions into 4 layers that each represents the key action
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
    