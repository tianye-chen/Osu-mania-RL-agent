import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image
from reward import Reward
from helper import detect, capture, pad_inner_array, SocketListener, DataQueue
from pynput import keyboard
from pynput.keyboard import Controller, Key
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import mss
import time
import pathlib
import os
import json
import random
import logging
import warnings
import sys
from model import ReplayMemory

class A2C_Agent():
  def __init__(self,
                ac_net,
                env,
                optimizer,
                max_episode,
                behavior_cloning = True,
                gamma=0.99,
                beta=0.5,
                grad_clip = 100.0,
              ):
    self.ac_net = ac_net
    self.env = env
    self.optimizer = optimizer
    self.max_episode = max_episode
    self.gamma = gamma
    self.beta = beta
    self.grad_clip = grad_clip
    
    self.memory = ReplayMemory(10000)
    self.expert_memory = ReplayMemory()
    self.a = []
    self.b = []
    
    if behavior_cloning:
      self._get_expert_replay()
  
  def train(self):
    total_rewards = []
    self.ac_net.train()
    
    for episode in range(self.max_episode):
      terminated = False
      states, multi_actions, rewards = [], [], []
      hx = None

      state = torch.tensor(self.env.reset(), dtype=torch.float32)
      while not terminated:
        probs, value, hx = self.ac_net(state, hx)
        action = []

        # Sample action for each lane based on probability distribution
        for prob in probs:
          prob = F.softmax(prob, dim=-1).squeeze()
          action.append(torch.distributions.Categorical(prob).sample().item())

        next_state, reward, terminated, _ = self.env.step(action)
          
        states.append(state)
        rewards.append(reward)
        multi_actions.append(action)

        state = torch.tensor(next_state, dtype=torch.float32)
        
      R = 0 if terminated else value

      # Calculate policy loss
      for i in reversed(range(len(rewards))):
        R = torch.tensor(rewards[i], dtype=torch.float32) + self.gamma * R
        R = R.detach()

        probs, value, hx = self.ac_net(states[i], hx)
        
        advantage = R - value
        policy_loss = []
        
        for i_a, prob in enumerate(probs):
          actions = multi_actions[i]
          softmax_probs = F.softmax(prob, dim=-1).squeeze()

          # Entropy value controlled by a decaying beta for exploration
          categorical_dist = torch.distributions.Categorical(softmax_probs)
          entropy = categorical_dist.entropy().sum() * max(0.001, self.beta - episode / self.max_episode)

          log_probs = -torch.log(softmax_probs)
          policy_loss.append(log_probs[actions[i_a]] * advantage - entropy)

      value_loss = F.mse_loss(value, R)

      loss = torch.sum(torch.stack(policy_loss)) + value_loss 
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_value_(self.ac_net.parameters(), self.grad_clip)
      self.optimizer.step()

      total_rewards.append(sum(rewards))
      meta_data = self.env.get_meta_data()
      print(f'Episode {episode:<5} {"[" + meta_data["song_name"] + "] " + str(meta_data["difficulty"]):<50} Reward: {sum(rewards):>10.4f}, loss: {loss.item():>10.4f}')
      print(self.env.reward_func.get_debug(self.env.render_mode))
      print(self.env.total_invalid_actions)
      print(self.env.actions_taken)
      
    return total_rewards
  
  def test(self):
    total_rewards = []
    hx = None

    for episode in range(self.max_episode):
      terminated = False
      truncated = False
      states, multi_actions, rewards = [], [], []

      state = torch.tensor(self.env.reset(), dtype=torch.float32)
      while not terminated:
        probs, value, hx = self.ac_net(state)
        action = []

        # Sample action for each lane with the highest probability
        for prob in probs:
          prob = F.softmax(prob, dim=-1).squeeze()
          action.append(torch.argmax(prob).item())

        next_state, reward, terminated, _ = self.env.step(action)

        states.append(state)
        rewards.append(reward)
        multi_actions.append(action)

        state = torch.tensor(next_state, dtype=torch.float32)

      total_rewards.append(sum(rewards))
      meta_data = self.env.get_meta_data()
      
      print(f'Episode {episode:<5} {"[" + meta_data["song_name"] + "] " + str(meta_data["difficulty"]):<50} Reward: {sum(rewards):>10.4f}')
      print(self.env.reward_func.get_debug())
      print(self.env.total_invalid_actions)
      print(self.env.actions_taken)
      
    return total_rewards
  
  def _get_expert_replay(self):
    path_name = './expert_demo'
    
    if os.path.exists(path_name):
      for file in os.listdir(path_name):
        if file.endswith('.pth'):
          replay = torch.load(os.path.join(path_name, file))
          self._update_expert_memory(replay) 
          
  def _update_expert_memory(self, replay):
    frames = replay['frame']
    actions = replay['action']
    hit_types = replay['hit_type']
    
    note_vector = [[0,0,0]] * self.env.max_notes
    state = deque([note_vector] * self.env.stacked_frames, maxlen=self.env.stacked_frames)
    holds = 0
    regulars = 0
    
    for i in range(len(frames)):
      note_vector = frames[i][:self.env.max_notes]
      note_vector = pad_inner_array([note_vector], [0, 0, 0], self.env.max_notes)[0]
      
      if note_vector != []:
        for n in note_vector:
          if n[0] == 1:
            holds += 1
          if n[0] == 2:
            regulars += 1
          
    self.a.append(holds) 
    self.b.append(regulars)
    print(holds, regulars)
    print(sum(self.a), sum(self.b))