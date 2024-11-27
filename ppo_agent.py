from model import ReplayMemory
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pynput.keyboard import Controller
import os
from collections import deque

class PPO_Agent:
    def __init__(self, 
                 env,
                 policy_net,
                 optimizer, 
                 discount_factor=0.99, 
                 batch_size=128, 
                 capacity=10000,
                 clip_epsilon=0.2
                 ):
        
        #define the environment
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.keyboard = Controller()

        # define hyperparameter
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.capacity = capacity
        self.clip_epsilon = clip_epsilon

        # define PPO
        self.policy_net = policy_net
        self.memory = ReplayMemory(5000)

        # keep track of trainning data
        self.rewards = []
        self.loss = []
        self.steps = []
        self.episode = 0