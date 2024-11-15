from utils import DQN, ReplayMemory
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
from pynput.keyboard import Controller

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
        self.keyboard = Controller()
        
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
        input_size = self.observation_space.shape[0]
        self.max_notes = self.observation_space.shape[1]
        self.policy_net = DQN(input_size, self.max_notes, self.action_space.nvec, dueling=dueling_dqn, noisy=noisy_dqn, std_init=std_init).to(device)
        self.target_net = DQN(input_size, self.max_notes. self.action_space.nvec, dueling=dueling_dqn, noisy=noisy_dqn, std_init=std_init).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.experience_replay = ReplayMemory(self.capacity)
        self.expert_replay = None # store expert replay for demonstration

        # track reward and loss
        self.rewards = []
        self.loss = []
        self.steps = []

    def _action_policy(self, state):
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
    
    def _update_model(self):
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
        q_values = self.policy_net(state_batch).gather(2, action_batch.unsqueeze(2)).squeeze(2)

        # compute expect q values
        next_state_values = torch.zeros_like(q_values, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=2).values

        expected_q_values = (next_state_values * self.discount_factor) + reward_batch

        # compute lose
        loss = self.criterion(q_values, expected_q_values)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

        self.optimizer.step()

        # soft update for target network
        self._soft_update()

        return loss.item()
    
    def _smooth_data(self, data):
        window_size = int(len(data)*0.05)
        window_size = max(1, window_size)
        return pd.Series(data).rolling(window=window_size).mean()
    
    def plot(self):
        average_rewards = self._smooth_data(self.rewards)
        plt.plot(average_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Rewards Over Episodes')
        plt.show()

        average_loss = self._smooth_data(self.loss)
        plt.plot(average_loss)
        plt.xlabel('Episode')
        plt.ylabel('Loss Value')
        plt.title('Loss Values Over Episodes')
        plt.show()
    
    def saveModel(self, name):
        torch.save(self.policy_net.state_dict(), f"{name}_network.pth")

        torch.save({
            'loss': self.loss,
            'reward': self.rewards,
            'step' : self.steps
            }, f"{name}_metadata.pth"
        )

    def loadModel(self, name):
        self.policy_net.load_state_dict(torch.load(f"{name}_network.pth"))
        metadata = torch.load(f"{name}_metadata.pth")

        self.loss = metadata['loss']
        self.rewards = metadata['reward']
        self.steps = metadata['step']

    def _soft_update(self):
         # soft update of the target's weights
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * self.target_update_rate + target_state_dict[key] * (1-
                                    self.target_update_rate)
        self.target_net.load_state_dict(target_state_dict)

    def train(self, total_episode = 500):
        for episode in range(1, total_episode+1):
            total_loss = 0
            total_reward = 0
            total_step = 0
            done = False
            state = self.env.reset()
            # shape[stack, max notes, 3] to [1, stack, max notes, 3] to [1, stack, max notes * 3]
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.max_notes, -1)
            time_total = time.time()
            while not done:
                if self.env.song_begin():
                    time_start = time.time()
                    action = self.action_space(state)
                    next_state, reward, terminate, truncate = self.env.step(action)
                    
                    # convert to proper tensor shape
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.max_notes, -1)
                    reward = torch.tensor([reward], device=device)
                    
                    done = terminate or truncate

                    # push to experience replay
                    self.experience_replay.push((state, action, reward, next_state, done))

                    time_total += time.time() - time_start

                    # mini update during the song
                    if self.env.getUpdateSignal():
                        self.env.pause_game()
                        # sample and update 25 time
                        for _ in range(32):
                            loss = self._update_model() 
                            if loss is not None:
                                total_loss += loss

                        self.env.Updated()
                        self.env.resume_game()

                    state = next_state
                    total_step += 1
                    total_reward += reward
                
            # big update after the sond ends
            for _ in range(64):
                loss = self._update_model()
                if loss is not None:
                    total_loss += loss
        
            self.env.return_to_song_selection_after_song()

            avg_loss = total_loss/total_step

            if episode%10 == 0:
                print(f'Epsiode: {episode}: Total Reward: {total_reward}, Loss: {avg_loss}')
                print(f"Average running time: {time_total/total_step} per step")

            self.loss.append(total_loss)
            self.rewards.append(total_reward)
            self.steps.append(total_step)



    