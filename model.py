import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math
import numpy as np

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
        
class LSTM_Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=128, num_layer=1, dropout=0):
        super(LSTM_Actor, self).__init__()
        self.num_frame, self.max_notes, self.note_feature = observation_space.shape

        self.lstm = nn.LSTM(input_size=self.max_notes*self.note_feature, 
                            hidden_size=hidden_size, 
                            batch_first=True, 
                            num_layers=num_layer, 
                            dropout=dropout)

        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True)
        )

        self.actors = nn.ModuleList([
            nn.Linear(128, dim) for dim in action_space
        ])


    def forward(self, x, hidden_state=None, softmax=True):
        lstm_out, _ = self.lstm(x, hidden_state)
        lstm_out = lstm_out[:, -1, :]

        lstm_out = self.linear_layer(lstm_out)

        if softmax:
            logits = [F.softmax(actor(lstm_out), dim=-1).unsqueeze(1) for actor in self.actors]
        else:
            logits = [actor(lstm_out).unsqueeze(1) for actor in self.actors]
        logits = torch.cat(logits, dim=1)

        return logits, hidden_state
    
class LSTM_Critic(nn.Module):
    def __init__(self, observation_space, hidden_size=128, num_layer=1, dropout=0):
        super(LSTM_Critic, self).__init__()
        self.num_frame, self.max_notes, self.note_feature = observation_space.shape

        self.lstm = nn.LSTM(input_size=self.max_notes*self.note_feature, 
                            hidden_size=hidden_size, 
                            batch_first=True, 
                            num_layers=num_layer, 
                            dropout=dropout)

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x, hidden_state=None):
        lstm_out, _ = self.lstm(x, hidden_state)
        lstm_out = lstm_out[:, -1, :]

        value = self.critic(lstm_out)

        return value.squeeze(0), hidden_state
      
class ReplayMemory:
    def __init__(self, capacity=None):
        """
        Create a queue to store the memory
        args:
            capacity: defaule None, create a queue with maximum capcity length 
        """
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        push experience to memory 

        args:
            state: observation state
            action: action based on current state
            reward: reward based on the action taken
            next_state: state after performing action
            done: truncate or terminate
        """
        self.memory.append((state, action, reward, next_state, done))

    def ppo_push(self, state, action, prob, value, reward, done):
        """
        push exerience to memory use for proximal policy optimization

        agrs:
            state: observation state
            action: action based on current state
            prob: log probability of the action
            value: obervation state value
            reward: reward based on the action taken
            done: truncate or terminate
        """
        self.memory.append((state, action, prob, value, reward, done))

    def ppo_sample(self, batch_size: int):
        """
        generate mini batches based on all memory for proximal policy optimization

        args:
            batch_size: sample batches based on the size

        returns:
            bacthes: array of batches where it start and end
            state: array of states that is store in memory
            action: array of actions that is store in memory
            prob: array of prob that is store in memory
            value: array of value that is store in memory
            reward: array of reward that is store in memory
            done: array of done flag that is store in memory
        """
        n_sample = len(self.memory)
        batch_start = np.arange(0, n_sample, batch_size)
        indices = np.arange(n_sample, dtype=np.int32)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]

        state, action, prob, value, reward, done = zip(*self.memory)

        return batches, state, action, prob, value, reward, done

    def sample(self, batch_size:int):
        """
        randomly sample a batch from memory
        
        args:
            batch_size: number of experiences to sample

        returns:
            a list of randomly sampled experiences
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """
        returns:
            the length of memory
        """
        return(len(self.memory))
    
    def clear(self):
        """
        empty out the memory
        """
        self.memory.clear()
    