from utils import DQN, ReplayMemory
import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, 
                 env, 
                 criterion, 
                 optimizer, 
                 discount_factor=0.99, 
                 epsilon_decay=100, 
                 epsilon_start=0.95,
                 epsilon_end=0.10, 
                 target_update_rate=0.005, 
                 batch_size=128, 
                 capacity=10000,
                 dueling_dqn=False,
                 noisy_dqn=False,
                 behavior_cloning=False,
                 std_init=0.5):
        
        #define the environment
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # define hyperparameter
        self.criterion = criterion
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.capacity = capacity
        self.noisy_dqn = noisy_dqn
        self.epsilon_update = 0

        # define the dqn
        input_size = sum(self.observation_space.shape)
        self.policy_net = DQN(input_size, self.action_space.nvec, dueling=dueling_dqn, noisy=noisy_dqn, std_init=std_init).to(device)
        self.target_net = DQN(input_size, self.action_space.nvec, dueling=dueling_dqn, noisy=noisy_dqn, std_init=std_init).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.experience_replay = ReplayMemory(self.capacity)
        self.expert_replay = None # store expert replay for demonstration

        # track reward and loss
        self.rewards = []
        self.loss = []

    def action_policy(self, state):
        if self.noisy_dqn:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=2)
        else:
            sample = random.random()
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.epsilon_update/self.epsilon_decay)
            self.epsilon_update += 1
            if sample < epsilon:
                return torch.tensor(self.action_space.sample(), dtype=torch.long, device=device)
            else:
                with torch.no_grad():
                    return self.policy_net(state).argmax(dim=2)
    
    def update_model(self):
        if len(self.experience_replay) < self.batch_size:
            return
        
        transitions= self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        reward_batch = torch.cat(reward_batch)

        # only get the next state where it isn't terminate or truncate
        non_final_mask = torch.tensor([not done for done in done_batch], dtype=torch.bool, device=device)
        non_final_next_states = torch.cat([s for s, done in zip(next_state_batch, done_batch) if not done])

        # compute q values
        q_values = self.policy_net(state_batch).gather(2, action_batch.unsqueeze(2))

        # compute expect q values
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values