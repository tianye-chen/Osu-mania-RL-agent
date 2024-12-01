from model import ReplayMemory
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pynput.keyboard import Controller
import os
from collections import deque
from torch.distributions.categorical import Categorical
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO_Agent:
    def __init__(self, 
                 env,
                 actor_net,
                 critic_net,
                 actor_optimizer,
                 critic_optimizer, 
                 discount_factor=0.99, 
                 batch_size=128, 
                 policy_clip=0.2, 
                 gae_lambda=0.9,
                 n_epoch=7,
                 behavior_cloning = False
                 ):
        
        #define the environment
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_frame = self.env.num_frame
        self.max_notes = self.env.max_notes
        self.keyboard = Controller()

        # define hyperparameter
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epoch = n_epoch
        self.behavior_cloning = behavior_cloning

        # define PPO
        self.actor = actor_net
        self.critic = critic_net
        self.actor_hidden = None
        self.critic_hidden = None
        self.memory = ReplayMemory()
        self.expert_replay = ReplayMemory()

        # keep track of trainning and pretraining data
        self.rewards = []
        self.loss = []
        self.steps = []
        self.episode = 0
        self.loss_pretrain = []

        if self.behavior_cloning:
            self._get_expert_demo()

    def pretrain(self, total_episode=100):
        """
        pretrain the model using expert demonstraction 

        args:
            total_episode: how many time the agent will train
        """
        self.actor.train()
        self.critic.train()
        avg_loss = 0.0
        criterion = torch.nn.MultiMarginLoss()
        for episode in range(1, total_episode+1):
            batches, states, actions, _, _, _, _ = self.expert_replay.ppo_sample(self.batch_size)
            states = torch.concat(states)
            actions = torch.concat(actions)

            for batch in batches:
                state_batch = torch.tensor(states[batch], device=device)
                action_batch = torch.tensor(actions[batch], device=device)

                logit, _ = self.actor(state_batch, softmax=False)                
                
                loss = 0
                for i in range(logit.shape[1]):
                    logit_key = logit[:, i, :] # shape (batch, num_key)
                    action_key = action_batch[:, i] # shape (batch, )

                    loss += criterion(logit_key, action_key)

                loss /= logit.shape[1]
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

                avg_loss += loss.item()
            
            avg_loss /= len(batches)
            self.loss_pretrain.append(avg_loss)

            if episode%10 == 0:
                print(f"Episod {episode}, pre-training loss: {avg_loss}")

    def train(self, total_episode=500, c=0.02):
        """
        train the agent for x number of episode
        Open the ous game and go to the song selection scene
        it will start training by randonly pick a song within the song bank predefined and start training
        it will keep looping till it is finish training automically

        args:
            total_episode: how many time the agent will train
            c: entropy coefficient which controls the weight of entropy
        """
        self.actor.train()
        self.critic.train()
        for episode in range(1,total_episode+1):
            total_reward = 0
            total_step = 0
            done = False
            self.actor_hidden = None
            self.critic_hidden = None
            state = self.env.reset()
            self.env.pick_random_song()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
            while not done and self.env.checking_connection():
                if self.env.song_begin():
                    action, prob, value = self._action_policy(state)
                    next_state, reward, terminate, truncate, info = self.env.step(action)

                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
                    reward = torch.tensor([reward], device=device)
                    self.memory.ppo_push(state, action, prob, value, reward, done)

                    state = next_state
                    done = truncate or terminate
                    total_reward += reward.item()
                    total_step += 1

                if self.env.lost_connection():
                    break
            
            avg_loss = self._ppo_update(c)
            self.episode += 1

            if self.episode % 10 ==0:
                print(f"Episode {self.episode}: Total Reward: {total_reward}, Loss: {avg_loss}")

            self.loss.append(avg_loss)
            self.rewards.append(total_reward)
            self.steps.append(total_step)

            self.env.return_to_song_selection_after_song()

            if episode == total_episode:
                self.keyboard.type("Finish training")
                
    def eval(self, total_episode=10, pretrain=False):
        """
        if evaluate for training:
            evaluate the agent for x number of episode
            Open the ous game and go to the song selection scene
            it will start evaluating by randonly pick a song within the song bank predefined and start evaluating
            it will keep looping till it is finish evaluating automically
        
        if evaluate for pretraining:
            print out the accuarcy by comparing predicted with truth action based on the state

        args:
            total_episode: how many time the agent will evaluate
            pretrain: evaluate flag for pretrain or train
        """
        self.actor.eval()
        self.critic.eval()

        if not pretrain:
            total_rewards = []
            songs = []
            total_steps = []
            for episode in range(1, total_episode+1):
                done = False
                episode_reward = 0
                episode_steps = 0
                self.actor_hidden = None
                state = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
                self.env.pick_random_song()
                while not done and self.env.checking_connection():
                    with torch.no_grad():
                        if self.env.song_begin():
                                logit, self.actor_hidden = self.actor(state, self.actor_hidden)
                                dist = Categorical(logits=logit)
                                action = dist.probs.argmax(dim=-1)
                                next_state, reward, terminate, truncate, _ = self.env.step(action)
                                
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

        else:
            correct = 0.0
            batches, states, actions, _, _, _, _ = self.expert_replay.ppo_sample(self.batch_size)
            states = torch.concat(states)
            actions = torch.concat(actions)

            for batch in batches:
                state_batch = torch.tensor(states[batch], device=device)
                action_batch = torch.tensor(actions[batch], device=device)

                logit, _ = self.actor(state_batch)
                dist = Categorical(logits=logit)
                action = dist.probs.argmax(dim=-1)
                
                for i in range(len(batch)):
                    if torch.equal(action[i], action_batch[i]):   
                        correct += 1
            
            accuracy = correct/len(actions)

            print("Accuracy: ", accuracy)

    def plot(self, pretrain=False):
        """
        plot the training and pretrain data

        args:
            pretrain: flag for training or pretain data 
        """
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
        """
        save the network and metadata into model_result folder
        network file will have _network.pth ending
        metadata file will have _metadata.pth ending

        args:
            name: name of the file
            pretrain: flag for training or pretrain; if pretrain, it will have _pretrain.pth ending
        """
        # ensure the folder exist
        os.makedirs('./model_results', exist_ok=True)

        if pretrain:
            torch.save({"actor": self.actor.state_dict(),
                        "critic": self.critic.state_dict()}, f"./model_results/{name}_network_pretrain.pth")
            torch.save({
                'loss_pretrain': self.loss_pretrain
                }, f"./model_results/{name}_metadata_pretrain.pth"
            )

        else:
            torch.save({"actor": self.actor.state_dict(),
                        "critic": self.critic.state_dict()}, f"./model_results/{name}_network.pth")
            torch.save({
                'loss': self.loss,
                'reward': self.rewards,
                'step' : self.steps
                }, f"./model_results/{name}_metadata.pth"
            )

        print("Model Saved")

    def loadModel(self, name, pretrain=False):
        """
        load the model and metadata information from model_result folder
        automically use ending with _network_pretrain.pth, _metadata_pretrain.pth if pretrain, _network.pth and _metadata_pth

        args:
            name: name of the file
            pretrian: flag for training or pretraining
        """
        if pretrain:
            network = torch.load(f"./model_results/{name}_network_pretrain.pth")
            self.actor.load_state_dict(network["actor"])
            self.critic.load_state_dict(network["critic"])
        
            metadata = torch.load(f"./model_results/{name}_metadata_pretrain.pth")
            self.loss_pretrain = metadata["loss_pretrain"]
        else:
            network = torch.load(f"./model_results/{name}_network.pth")
            self.actor.load_state_dict(network["actor"])
            self.critic.load_state_dict(network["critic"])
        
            metadata = torch.load(f"./model_results/{name}_metadata.pth")

            self.loss = metadata['loss']
            self.rewards = metadata['reward']
            self.steps = metadata['step']
            self.episode = len(self.loss)

        print("Model Loaded")

    def _ppo_update(self, c=0.2, expert_weight=0.2) -> float:
        """
        Update the gradient by n number of epoch from mini batches 
        combine the loss with ppo loss and expert loss

        args:
            c: entropy coefficient which control exploration rate
            expert_weight: weights for the expert loss
        
        returns:
            avg_loss: average loss 
        """
        avg_loss = 0.0
        criterion = torch.nn.MultiMarginLoss()
        for _ in range(self.n_epoch):
            batches, states, actions, old_probs, values, rewards, dones = self.memory.ppo_sample(self.batch_size)
            _, expert_states, expert_actions, _, _, _, _ = self.expert_replay.ppo_sample(self.batch_size)

            states = torch.concat(states)
            actions = torch.concat(actions)
            old_probs = torch.concat(old_probs)
            values = torch.concat(values)
            rewards = torch.concat(rewards)
            dones = torch.tensor([done for done in dones], dtype=torch.long, device=device)

            expert_states = torch.concat(expert_states)
            expert_actions = torch.concat(expert_actions)

            advantages = self._compute_gae(rewards, values, dones)

            for batch in batches:
                state_batch = torch.tensor(states[batch], device=device)
                old_prob_batch = torch.tensor(old_probs[batch], device=device)
                action_batch = torch.tensor(actions[batch], device=device)

                expert_state_batch = torch.tensor(expert_states[batch], device=device)
                expert_action_batch = torch.tensor(expert_actions[batch], device=device)

                logit, _ = self.actor(state_batch)
                dist = Categorical(logits=logit)
                value, _ = self.critic(state_batch)
                new_prob = dist.log_prob(action_batch)
                entroy = dist.entropy().mean()
                
                ratio = new_prob.exp()/old_prob_batch.exp()
                return_batch = advantages[batch] + value
                sur1 = advantages[batch] * ratio
                sur2 = torch.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip)*advantages[batch]
                actor_loss = -torch.min(sur1, sur2).mean()
                critic_loss = torch.nn.MSELoss()(value, return_batch)

                expert_loss = 0
                expert_logit, _ = self.actor(expert_state_batch)
                for i in range(logit.shape[1]):
                    logit_key = expert_logit[:, i, :] # shape (batch, num_key)
                    action_key = expert_action_batch[:, i] # shape (batch, )

                    expert_loss += criterion(logit_key, action_key)

                expert_loss /= logit.shape[1]

                loss = actor_loss + 0.5 * critic_loss - c * entroy + expert_weight*expert_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                avg_loss += loss.item()
            
            avg_loss /= len(batches)
                
        avg_loss /= self.n_epoch
        self.memory.clear()

        return avg_loss

    def _action_policy(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Select action based on the observation state

        args:
            state: observation state of shape (1, state_dim)

        returns:
            action: selected action of shape (1, num_keys)
            probs: log probability of the action of shape (1, num_keys)
            value:  obervation state value (1,)
        '''
        with torch.no_grad():
            logit, self.actor_hidden = self.actor(state, self.actor_hidden)
            value, self.critic_hidden = self.critic(state, self.critic_hidden)

            dist = Categorical(logits=logit)
            action = dist.sample()
            probs = dist.log_prob(action)

        return action, probs, value

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        '''
        Compute generalized advantage estimation (GAE)

        args: 
            rewards: reward of shape (batch,)
            values: state value of shape (batch,)
            dones: done flags of shape (batch,)

        returns:
            advanatage: computed Advantages of shape (batch, 1)
        '''
        advantage = torch.zeros(len(rewards), device=device, dtype=torch.float32)
        gae = 0
        next_value = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.discount_factor * next_value * (1-dones[i]) - values[i]
            gae = delta + self.discount_factor - self.gae_lambda * (1-dones[i]) * gae
            advantage[i] = gae
            next_value = values[i]

        advantage = (advantage - advantage.mean()) / (advantage.std()+1e-10)

        return advantage.unsqueeze(1)
    
    def _smooth_data(self, data):
        """
        smoothen the data for plotting 

        args:
            data: data for smoothing

        returns:
            data: smoothen data
        """
        window_size = int(len(data)*0.05)
        window_size = max(1, window_size)
        return pd.Series(data).rolling(window=window_size).mean()
    
    def _get_expert_demo(self):
        """
        loop throught all file within the expert demo folder and push into the memory
        """
        folder_path = "./expert_demo"
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pth"):
                file_path = os.path.join(folder_path, file_name)
                replay = torch.load(file_path)
                self._update_expert_replay(replay)

    def _update_expert_replay(self, replay):
        """
        push the replay into the memory

        args:
            replay: a batch of frames that represents the demo of that song
        """
        frames = replay["frame"]
        actions = replay["action"]
        hit_types = replay["hit_type"]

        note_vector = [[0,0,0]] * self.max_notes
        state = deque(maxlen=self.num_frame)

        for _ in range(self.num_frame):
            state.append(note_vector)

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

            action = actions[i]

            hit_type = [hit_types[i]]
            reward, truncate, terminate = 0, False, False
            if hit_type is not None:
                reward, truncate, terminate = self.env._get_reward(hit_type)
            done = truncate or terminate
    
            # convert to tensor
            _state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, self.num_frame, -1)
            _action = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(0)
            _reward = torch.tensor([reward], device=device)
            
            self.expert_replay.ppo_push(_state, _action, 0, 0, _reward, done)
