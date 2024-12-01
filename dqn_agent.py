from model import ReplayMemory
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pynput.keyboard import Controller
import os
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_Agent:
    def __init__(self, 
                 env,
                 policy_net,
                 target_net,
                 optimizer, 
                 discount_factor=0.99, 
                 epsilon_decay=100, 
                 epsilon_start=0.95,
                 epsilon_end=0.10, 
                 target_update_rate=0.005, 
                 batch_size=128, 
                 capacity=10000,
                 lstm = False, 
                 behavior_cloning = False):
        
        #define the environment
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.max_notes = self.env.max_notes
        self.num_frame = self.env.num_frame
        self.keyboard = Controller()
        
        # define hyperparameter
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.capacity = capacity
        self.behavior_cloning = behavior_cloning

        # define the dqn
        self.policy_net = policy_net
        self.target_net = target_net
        self.experience_replay = ReplayMemory(self.capacity)
        self.expert_replay = ReplayMemory()
        self.lstm = lstm
        self.hidden = None

        # keep track of trainning data
        self.rewards = []
        self.loss = []
        self.steps = []
        self.epsilon_update = 0
        self.episode = 0

        # keep track of pretraining data
        self.loss_pretrain = []

        if self.behavior_cloning:
            self._get_expert_demo()

    def pretrain(self, margin=0.8, total_episode=1000):
        self.policy_net.train()
        for episode in range(1, total_episode+1):
            loss = self._update_expert_model(margin)

            self.loss_pretrain.append(loss.item())

            if episode%10 == 0:
                print(f"Episod {episode}, pre-training loss: {loss.item()}")

    def train(self, margin=0.1, total_episode = 500):
        self.policy_net.train()
        for episode in range(1, total_episode+1):
            total_loss = 0
            total_reward = 0
            total_step = 0
            update_count = 0
            done = False
            self.hidden = None
            state = self.env.reset()
            self.env.pick_random_song()
            # shape[stack, max notes, 3] to [1, stack, max notes, 3] to [1, stack, max notes * 3]
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
            while not done and self.env.checking_connection():
                if self.env.song_begin():
                    action = self._action_policy(state)
                    next_state, reward, terminate, truncate, info = self.env.step(action)
                    
                    # convert to proper tensor shape
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
                    total_reward += reward
                    reward = torch.tensor([reward], device=device)
                    
                    done = terminate or truncate

                    # push to experience replay where state is not all zeros
                    if not torch.all(state==0):
                        if not info["idle"]  or random.random() < 0.1: # avoid memory to overflow with unimportant state
                            self.experience_replay.push(state, action, reward, next_state, done)
                            update_count += 1

                    state = next_state
                    total_step += 1

                # break when socket connection disconnect
                if self.env.lost_connection():
                    break
                
            # update after the sond ends
            count = 0
            for _ in range(int(update_count/2)):
                loss = self._update_model(margin)
                if loss is not None:
                    total_loss += loss.item()
                    count += 1

            avg_loss = total_loss/count if count > 0 else 0
            self.episode += 1
            self.epsilon_update += 1

            if self.episode%10 == 0:
                print(f'Epsiode: {self.episode}: Total Reward: {total_reward}, Loss: {avg_loss}')

            self.loss.append(avg_loss)
            self.rewards.append(total_reward)
            self.steps.append(total_step)

            self.env.return_to_song_selection_after_song()

            if episode == total_episode:
                self.keyboard.type("Finish training")

    def eval(self, total_episode):
        self.policy_net.eval()
        total_rewards = []
        songs = []
        total_steps = []
        self.hidden = None
        for episode in range(1, total_episode+1):
            done = False
            episode_reward = 0
            episode_steps = 0
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
            self.env.pick_random_song()
            while not done and self.env.checking_connection():
                with torch.no_grad():
                    if self.env.song_begin():
                            if self.lstm:
                                q_values, self.hidden = self.policy_net(state, self.hidden)
                                action = q_values.argmax(dim=2)
                            else:
                                action = self.policy_net(state).argmax(dim=2)

                            next_state, reward, terminate, truncate, _ = self.env.step(action, train=False)
                            
                            # convert to proper tensor shape
                            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
                            done = terminate or truncate
                            state = next_state
                            episode_reward += reward
                            episode_steps += 1

                    # break when socket connection disconnect or timeout 
                    if self.env.lost_connection():
                        break
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
            songs.append(self.env.getSong())

            self.env.return_to_song_selection_after_song()

            if episode == total_episode:
                self.keyboard.type("Finish evaluating")

        plt.plot(songs, total_rewards)
        plt.xlabel('Song Name')
        plt.ylabel('Reward')
        plt.title('Rewards Over Each Song')
        plt.xticks(rotation=45, ha='right')
        plt.show()

        plt.plot(songs, total_steps)
        plt.xlabel('Song Name')
        plt.ylabel('Step')
        plt.title('Steps Over Each Song')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def plot(self, pretrain=False):
        if not pretrain:
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

            average_steps = self._smooth_data(self.steps)
            plt.plot(average_steps)
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.title('Steps Over Episodes')
            plt.show()

        else:
            average_loss = self._smooth_data(self.loss_pretrain)
            plt.plot(average_loss)
            plt.xlabel('Episode')
            plt.ylabel('Loss Value')
            plt.title('Loss Values Over Episodes')
            plt.show()

    def saveModel(self, name, pretrain=False):
        # ensure the folder exist
        os.makedirs('./model_results', exist_ok=True)
        if pretrain:
            torch.save(self.policy_net.state_dict(), f"./model_results/{name}_network_pretrain.pth")
            torch.save({
                'loss_pretrain': self.loss_pretrain
                }, f"./model_results/{name}_metadata_pretrain.pth"
            )

        else:
            torch.save(self.policy_net.state_dict(), f"./model_results/{name}_network.pth")
            torch.save({
                'loss': self.loss,
                'reward': self.rewards,
                'step' : self.steps
                }, f"./model_results/{name}_metadata.pth"
            )

        print("Model Saved")

    def loadModel(self, name, pretrain=False):
        if pretrain:
            self.policy_net.load_state_dict(torch.load(f"./model_results/{name}_network_pretrain.pth"))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
            metadata = torch.load(f"./model_results/{name}_metadata_pretrain.pth")
            self.loss_pretrain = metadata["loss_pretrain"]
        else:
            self.policy_net.load_state_dict(torch.load(f"./model_results/{name}_network.pth"))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
            metadata = torch.load(f"./model_results/{name}_metadata.pth")

            self.loss = metadata['loss']
            self.rewards = metadata['reward']
            self.steps = metadata['step']

        print("Model Loaded")

    def _action_policy(self, state):
        with torch.no_grad():
            if self.lstm:
                q_values, self.hidden = self.policy_net(state, self.hidden)
                action = q_values.argmax(dim=2)
            else:
                action = self.policy_net(state).argmax(dim=2)

        sample = random.random()
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.epsilon_update/self.epsilon_decay)
        if sample < epsilon:
            return torch.tensor(self.action_space.sample(), dtype=torch.long, device=device).unsqueeze(0)
        else:
            return action
    
    def _get_expert_demo(self):
        folder_path = "./expert_demo"
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pth"):
                file_path = os.path.join(folder_path, file_name)
                replay = torch.load(file_path)
                self._update_expert_replay(replay)

    def _update_expert_replay(self, replay):
        frames = replay["frame"]
        actions = replay["action"]
        hit_types = replay["hit_type"]

        note_vector = [[0,0,0]] * self.max_notes
        state = deque(maxlen=self.num_frame)
        next_state = deque(maxlen=self.num_frame)

        for _ in range(self.num_frame):
            state.append(note_vector)
            next_state.append(note_vector)

        for i in range(len(frames)):
            note_vector = frames[i] 
            note_vector = note_vector[:self.max_notes]
            if note_vector != []:
                note_vector = np.array(note_vector, dtype=np.float32)
                note_vector[:, :2] += 1
                note_vector = note_vector.tolist()
                
            note_vector += [[0,0,0]] * (self.max_notes - len(note_vector))
            note_vector = np.array(note_vector, dtype=np.float32)
            note_vector[:, 2] /= 100.0

            state.append(note_vector)
            next_state.append(note_vector)

            action = actions[i]

            hit_type = hit_types[i]
            reward, truncate, terminate = 0, False, False
            if hit_type is not None:
                reward, truncate, terminate = self.env._get_reward(hit_type)
            done = truncate or terminate

            # store for next state
            if i + 1 < len(frames):
                note_vector = frames[i+1] 
                note_vector = note_vector[:self.max_notes]
                if note_vector != []:
                    note_vector = np.array(note_vector, dtype=np.float32)
                    note_vector[:, :2] += 1
                    note_vector = note_vector.tolist()

                note_vector += [[0,0,0]] * (self.max_notes - len(note_vector))
                note_vector = np.array(note_vector, dtype=np.float32)
                note_vector[:, 2] /= 100.0

                next_state.append(note_vector)
            else:
                note_vector = [[0,0,0]] * self.max_notes
                next_state.append(note_vector)
    
            # convert to tensor
            _state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
            _action = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(0)
            _reward = torch.tensor([reward], device=device)
            _next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)

            if not torch.all(_state==0) or random.random() < 0.5:
                if reward != 0 or random.random() < 0.1:
                    self.expert_replay.push(_state, _action, _reward, _next_state, done)

    def _update_expert_model(self, margin=0.8, expert_sample=1, pretrain=True):
        td_loss, q_values, expert_action = self._compute_td_loss(expert_data=True, sample_ratio=expert_sample)

        margin_values = torch.full_like(q_values, margin, device=device, dtype=torch.float32)

        batch_indices = torch.arange(q_values.shape[0], device=device).unsqueeze(1)
        action_indices = torch.arange(expert_action.shape[1], device=device)

        margin_values[batch_indices, action_indices, expert_action] = 0

        # Q(s,a) + I(a_expect, a) - Q(s, a_expert)
        margin_loss = torch.max((q_values + margin_values), dim=2)[0] - q_values.gather(2, expert_action.unsqueeze(2)).squeeze(2)

        # include td loss and margin loss 
        loss = td_loss + margin_loss.mean()

        if pretrain:
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            if not self.lstm:
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            else:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5)

            self.optimizer.step()

            # soft update for target network
            self._soft_update()

        return loss 

    def _update_model(self, margin=0.1):
        self_sample = 1
        if len(self.experience_replay) < self.batch_size:
            return
        
        # weights loss and sample % for expert and self-generated data using epsilon decay
        if self.behavior_cloning:
            expert_weight = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.epsilon_update/self.epsilon_decay)
            self_weight = 1 - expert_weight

            expert_sample = 0.2 + (0.8-0.2) * np.exp(-1 * self.epsilon_update/self.epsilon_decay)
            self.sample = 1 - expert_sample

        td_loss, _, _ = self._compute_td_loss(expert_data=False, sample_ratio=self_sample)

        if self.behavior_cloning:
            expert_loss = self._update_expert_model(margin, expert_sample, pretrain=False)
            loss = self_weight*td_loss + expert_weight*expert_loss
        else:
            loss = td_loss

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        if not self.lstm:
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        else:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5)

        self.optimizer.step()

        # soft update for target network
        self._soft_update()

        return loss
    
    def _smooth_data(self, data):
        window_size = int(len(data)*0.05)
        window_size = max(1, window_size)
        return pd.Series(data).rolling(window=window_size).mean()
    
    def _compute_td_loss(self, expert_data=False, sample_ratio=1):
        if expert_data:
            transitions= self.expert_replay.sample(int(sample_ratio*self.batch_size))
        else:
            transitions = self.experience_replay.sample(int(sample_ratio*self.batch_size))

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        reward_batch = torch.cat(reward_batch)
        next_state_batch = torch.cat(next_state_batch)
        done_batch= torch.tensor([done for done in done_batch], dtype=torch.long, device=device)
       
        # compute q values
        if self.lstm:
            q_values, _ = self.policy_net(state_batch)
        else:
            q_values = self.policy_net(state_batch)

        q_values_optimal = q_values.gather(2, action_batch.unsqueeze(2)).squeeze(2)

        # compute expect q values
        with torch.no_grad():
            if self.lstm:
                next_q_values, _ = self.target_net(next_state_batch)
                
            else:
                next_q_values = self.target_net(next_state_batch)

            next_q_values_optimal = next_q_values.max(dim=2).values
                
        expected_q_values = (1-done_batch.unsqueeze(1)) * (next_q_values_optimal * self.discount_factor) + reward_batch.unsqueeze(1)

        # compute td loss using mse
        loss = self.criterion(q_values_optimal, expected_q_values)

        return loss, q_values, action_batch
    
    def _soft_update(self):
         # soft update of the target's weights
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * self.target_update_rate + target_state_dict[key] * (1-
                                    self.target_update_rate)
        self.target_net.load_state_dict(target_state_dict)