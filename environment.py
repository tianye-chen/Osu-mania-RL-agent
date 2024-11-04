import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pynput.keyboard import Controller
from concurrent.futures import ThreadPoolExecutor

class OsuEnvironment(gym.env):
    def __init__(self):
        # the keys corresponading to the 4 lane
        self.keys = ['s', 'd', 'k', 'l']
        # dyanamic list of vectors return tuple of (type, lane, y position) 
        # placehold for fixed size as 20
        self.observation = spaces.Box(low=0, high=np.lnf, shape=(20, 3), dtype=np.float32)
        # 4 lane where each note can do nothing, pressed, held or release
        self.action_space = spaces.MultiDiscrete([4]*len(self.keys))
        # keep track of which key is hold
        self.currently_hold = [False] * len(self.keys)
        # check for invalid key press
        self.invalid = False
        self.keyboard = Controller()

    def reset(self):
        self.currently_hold = [False] * len(self.keys)
        self.observation = np.empty()
        self.invalid = False
        return self.observation
    
    def keyboard_action(self, key, action):
        match action:
            case 0: # do nothing
                return
            case 1: # press
                self.keyboard.press(key)
                self.keyboard.release(key)
            case 2: # hold
                self.keyboard.pressed(key)
                self.currently_hold[key] = True
            case 3: # release
                if self.currently_hold[key]:
                    self.keyboard.release(key)
                else:
                    self.invalid = True

    def step(self, actions):
        # take action based on the given actions simultaneously
        with ThreadPoolExecutor(max_workers=len(self.keys)) as executor:
            for lane in range(actions):
                executor.submit(self.keyboard_action(self.keys[lane], actions[lane]))
        
        # some method to get socket
        socket = getsocket()

        # reward based on the action taken
        match socket:
            case 0: # miss
                reward = -3
            case 1: # meh
                reward = -2
            case 2: # ok
                rewrad = -1
            case 3: # good
                reward = 1
            case 4: # great
                reward = 2
            case 5: # perfect
                reward = 3

        # if there are invalid action
        if self.invalid:
            reward += -10

        self.invalid = False

        # some method to get the notes vector, truncate and terminate state
        self.observation = getNotes()
        truncate = getTruncate()
        terminate = getTerminate()
        
        return self.observation, reward, truncate, terminate