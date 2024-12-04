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
                batch_size=64,
                gamma=0.9,
                beta=0.5,
                grad_clip = 1.0,
                device='cpu'
              ):
    self.ac_net = ac_net
    self.env = env
    self.optimizer = optimizer
    self.gamma = gamma
    self.beta = beta
    self.device = device
    self.grad_clip = grad_clip
    self.batch_size = batch_size
    self.pretrain_accuracies = []
    self.pretrain_losses = []
    
    self.memory = ReplayMemory(10000)
    self.expert_memory = ReplayMemory()
    
    # Debug
    self.action_totals = [0] * 4
    self.hit_type_totals = [0] * 8

    self._get_expert_replay()
      
  def pretrain(self, max_episode):
    self.ac_net.train()
    criterion = nn.CrossEntropyLoss()
    
    for episode in range(max_episode):
      correct = 0
      total = 0
      total_loss = 0
      batches, states, actions, *_ = self.expert_memory.ppo_sample(self.batch_size)
      states = torch.stack(states)
      actions = torch.stack(actions)

      for batch in batches:
        state_batch = torch.tensor(states[batch], device=self.device)
        action_batch = torch.tensor(actions[batch], dtype=torch.long, device=self.device)
        
        probs, _, _ = self.ac_net(state_batch)                
        
        loss = 0
        for i in range(probs.shape[1]):
          lane_action = action_batch[:, i]
          lane_probs = probs[:, i, :]
          loss += criterion(lane_probs, lane_action)
          
        loss /= 4
        self.pretrain_losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
        nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.grad_clip)
        
        probs_softmax = F.softmax(probs, dim=-1).detach()
        preds = torch.argmax(probs_softmax, dim=-1)
        
        for p, a in zip(preds, action_batch):
          total += 1
          if all(p == a):
            correct += 1
            
      self.pretrain_accuracies.append(correct / total)
      
      if np.mean(self.pretrain_accuracies[-3:]) > 0.98:
        break
      
      print(f'Episode {episode:<5} Loss: {total_loss/len(batches):>10.4f}, Accuracy: {np.mean(self.pretrain_accuracies[-5:]):>10.4f}')
  
  def train(self, max_episode):
    total_rewards = []
    self.ac_net.train()
    criterion = nn.CrossEntropyLoss()
    
    for episode in range(max_episode):
      terminated = False
      states, multi_actions, rewards = [], [], []
      hx = None

      state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
      while not terminated:
        probs, value, hx = self.ac_net(state, hx)
        probs = F.softmax(probs, dim=-1).squeeze()
        action = []

        # Sample action for each lane based on probability distribution
        for prob in probs:
          action.append(torch.distributions.Categorical(prob).sample().item())

        next_state, reward, terminated, _ = self.env.step(action)
          
        states.append(state)
        rewards.append(reward)
        multi_actions.append(action)

        state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
      R = 0 if terminated else value

      # Calculate policy loss
      for i in reversed(range(len(rewards))):
        R = torch.tensor(rewards[i], dtype=torch.float32, device=self.device) + self.gamma * R
        R = R.detach()

        probs, value, hx = self.ac_net(states[i], hx)
        
        advantage = R - value
        policy_loss = []
        
        for i_a, prob in enumerate(probs):
          actions = multi_actions[i]
          softmax_probs = F.softmax(prob, dim=-1).squeeze()

          # Entropy value controlled by a decaying beta for exploration
          categorical_dist = torch.distributions.Categorical(softmax_probs)
          entropy = categorical_dist.entropy().mean() * max(0.001, self.beta - episode / max_episode)

          log_probs = -torch.log(softmax_probs)
          policy_loss.append(log_probs[actions[i_a]] * advantage - entropy)
          
      value_loss = F.mse_loss(value, R)
      
      # Calculate expert loss based on policy network actions compared to expert actions
      expert_experiences = self.expert_memory.sample(len(rewards))
      expert_states, expert_actions, *_ = zip(*expert_experiences)
      expert_states = torch.stack(expert_states)
      expert_actions = torch.stack(expert_actions).long()
      
      expert_probs, _, _ = self.ac_net(expert_states)
      expert_loss = 0
      
      for i in range(expert_probs.shape[1]):
        lane_action = expert_actions[:, i]
        lane_probs = expert_probs[:, i, :]
        expert_loss += criterion(lane_probs, lane_action)
      
      expert_loss /= 4

      loss = torch.sum(torch.stack(policy_loss)) + value_loss * 0.5 + expert_loss * 0.2
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.grad_clip)
      self.optimizer.step()

      total_rewards.append(sum(rewards))
      meta_data = self.env.get_meta_data()
      print(f'Actor loss: {torch.sum(torch.stack(policy_loss))}, Value loss: {value_loss * 0.5}, Expert loss: {expert_loss * 0.2}')
      print(f'Episode {episode:<5} {"[" + meta_data["song_name"] + "] " + str(meta_data["difficulty"]):<50} Reward: {sum(rewards):>10.4f}, loss: {loss.item():>10.4f}')
      print(self.env.reward_func.get_debug(self.env.render_mode))
      print(self.env.total_invalid_actions)
      print(self.env.actions_taken)
      
    return total_rewards
  
  def test(self, max_episode):
    self.ac_net.eval()
    total_rewards = []
    hx = None
    
    for episode in range(max_episode):
      terminated = False
      states, multi_actions, rewards = [], [], []

      state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
      while not terminated:
        probs, value, hx = self.ac_net(state, hx)
        probs = F.softmax(probs, dim=-1).squeeze()
        action = []
        
        # Sample action for each lane with the highest probability
        for prob in probs:
          action.append(torch.argmax(prob).item())

        next_state, reward, terminated, _ = self.env.step(action)

        states.append(state)
        rewards.append(reward)
        multi_actions.append(action)

        state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
      total_rewards.append(sum(rewards))
      meta_data = self.env.get_meta_data()
      
      print(f'Episode {episode:<5} {"[" + meta_data["song_name"] + "] " + str(meta_data["difficulty"]):<50} Reward: {sum(rewards):>10.4f}')
      print(self.env.reward_func.get_debug(self.env.render_mode))
      print(self.env.total_invalid_actions)
      print(self.env.actions_taken)
      
    return total_rewards
  
  def _get_expert_replay(self):
    '''
    Loads expert replay .pth files from ./expert_demo,
    demos should be created using the record_demo.ipynb notebook
    '''
    path_name = './expert_demo'
    
    if os.path.exists(path_name):
      for file in os.listdir(path_name):
        if file.endswith('.pth'):
          replay = torch.load(os.path.join(path_name, file))
          self._update_expert_memory(replay) 
            
    print(f'Found {self.action_totals} actions')
    print(f'Found hit types Miss: {self.hit_type_totals[0]}, Meh: {self.hit_type_totals[1]}, Ok: {self.hit_type_totals[2]}, ' +
          f'Good: {self.hit_type_totals[3]}, Great: {self.hit_type_totals[4]}, Perfect: {self.hit_type_totals[5]}, ' + 
          f'Passed: {self.hit_type_totals[6]}, Failed: {self.hit_type_totals[7]}')
          
  def _update_expert_memory(self, replay):
    '''
    Extracts, processes and pushes replay experiences to expert memory
    '''
    frames, actions, hit_types = replay.values()
    
    # Rolling window of n stacked frames initialized with empty notes
    note_vector = [[0,0,0]] * self.env.max_notes
    state = deque([note_vector] * self.env.stacked_frames, maxlen=self.env.stacked_frames)
    
    for i in range(len(frames)):
      note_vector = frames[i][:self.env.max_notes]
      note_vector = pad_inner_array([note_vector], [0, 0, 0], self.env.max_notes)[0]
      note_vector = np.array(note_vector, dtype=np.float32)
      note_vector[:, 2] = note_vector[:, 2] / 100.0 # Normalize y_center
      
      state.append(note_vector)
      action = actions[i]
      hit_type = hit_types[i]
      terminated = (6 in hit_type) or (7 in hit_type)
      reward = self.env.reward_func.get_in_game_reward(hit_type)
      
      _state = torch.tensor(state, dtype=torch.float32, device=self.device)
      _action = torch.tensor(action, dtype=torch.float32, device=self.device)
      _reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
      
      for h in hit_type:
        self.hit_type_totals[h] += 1

      # Experiences with more than two 0s in the action set are pushed to memory with a 50% chance
      #if not sum(_action == 0) > 2 or random.random() < 0.5:
      for a in _action:
        self.action_totals[int(a)] += 1
        
      self.expert_memory.ppo_push(_state, _action, 0, 0, _reward, terminated)
      