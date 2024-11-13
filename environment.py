import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pynput.keyboard import Controller
from concurrent.futures import ThreadPoolExecutor
from helper import SocketListener
import matplotlib.pyplot as plt
import torch
import mss
import pathlib
from collections import deque

class OsuEnvironment(gym.Env):
    def __init__(self):
        # set frame per second
        self.frame_interval = 1 / 60

        # setup the neccessary resources for vision task
        self._vision_setup()

        # the keys corresponading to the 4 lane
        self.keys = ['s', 'd', 'k', 'l']

        # stacked frames of notes
        self.observation = deque(maxlen=4)

        # returns 4 stacked of list of note vectors in [type, lane, y position]
        # maximum notes return is 12 
        self.max_notes = 12
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4, self.max_notes, 3), dtype=np.float32)

        # 4 lane where each note can do nothing, pressed, held or release
        self.action_space = spaces.MultiDiscrete([4]*len(self.keys))
        
        # keep track of which key is hold
        self.currently_hold = [False] * len(self.keys)

        # check for invalid key press
        self.invalid = False
        self.keyboard = Controller()
        
        # create a number of thread for use
        self.executor = ThreadPoolExecutor(max_workers=7)

        # socket setup
        self.listener = SocketListener()

    def reset(self):
        self.currently_hold = [False] * len(self.keys)
        self.invalid = False
        self.listener.start()
        self.observation = np.zeros((4, 12, 3), dtype=int)
        
    def checking_connection(self):
        return self.listener.is_listening or self.listener.is_first_connection
    
    def song_begin(self):
        return self.listener.has_connection
    

    def step(self, actions):
        reward = 0
        truncate = False
        terminate = False

        # detect notes
        self._notes_detection()

        # take action based on the given actions simultaneously and fetech the hit type after action
        data = self.listener.fetch_data(action_fuc=lambda: self._perform_action(actions), timeout=0.02)

        if data is not None:
            data = int.from_bytes(data, byteorder='little')
            reward, truncate, terminate = self._get_reward(data)

        return self.observation, reward, truncate, terminate

    def _keyboard_action(self, lane, key, action):
        match action:
            case 0: # do nothing
                if self.currently_hold[lane]:
                    self.invalid = True
                else:
                    return
            case 1: # press
                if self.currently_hold[lane]:
                    self.invalid = True
                else:
                    self.keyboard.press(key)
                    self.keyboard.release(key)
            case 2: # hold
                if self.currently_hold[lane]:
                    return
                self.keyboard.press(key)
                self.currently_hold[lane] = True
            case 3: # release
                if self.currently_hold[lane]:
                    self.keyboard.release(key)
                    self.currently_hold[lane] = False
                else:
                    self.invalid = True
            # debug
            # self.keyboard.press(key)
            # time.sleep(0.05)
            # self.keyboard.release(key)

    def _get_reward(self, hit_type):
        reward = 0 
        truncate = False
        terminate = False
        # reward based on the action taken
        match hit_type:
            case 0: # miss
                reward = -3
            case 1: # meh
                reward = -2
            case 2: # ok
                reward = -1
            case 3: # good
                reward = 1
            case 4: # great
                reward = 2
            case 5: # perfect
                reward = 3
            case 6: # pass
                terminate = True
                self.listener.stop()
            case 7: # failure
                truncate = True
                self.listener.stop()
                
        # if there are invalid action
        if self.invalid:
            reward += -10
            self.invalid = False

        return reward, truncate, terminate

    def _vision_setup(self):
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        pathlib.PosixPath = pathlib.WindowsPath # https://github.com/ultralytics/yolov5/issues/10240#issuecomment-1662573188
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt', force_reload=True)  

        self.monitor = mss.mss().monitors[1]
        t, l, w, h = self.monitor['top'], self.monitor['left'], self.monitor['width'], self.monitor['height']
        self.region = {'left': l+int(w * 0.338), 'top': t, 'width': w-int(w * 0.673), 'height': h} 
    
    def _detect(self, img, model):
        lanes = {
            0 : (10, 180),
            1 : (150, 320),
            2 : (300, 470),
            3 : (440, 610)
        }
        ret = []  
        res = model(img)
  
        for box in res.xyxy[0]:
            # Confidence level is less than 50%
            if box[4] < 0.50:
                continue
    
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
            class_id = int(box[5]) # classes are 0: end_hold, 1: note, 2: start_hold
        
            # Identify the lane of the note based on x_center
            for lane, (start, end) in lanes.items():
                if start <= x_center <= end:
                    break
        
            ret.append([class_id, lane, y_center])

        # only care about notes that are near the hit window
        ret = sorted(ret, key=lambda note: note[2])
        ret = ret[:self.max_notes]

        # add padding if it is less than max note
        ret += [[0,0,0]] * (self.max_notes - len(ret))
        return ret 

    def _capture(self,region):
        with mss.mss() as sct:
            return sct.grab(region)
        
    def _notes_detection(self):
        vision_thread = self.executor.submit(self._capture, self.region)
        image = vision_thread.result()

        vision_thread = self.executor.submit(self._detect, np.array(image), self.model)
        self.observation = vision_thread.result()

    def _perform_action(self, actions):
        for lane in range(len(actions)):
            self.executor.submit(self._keyboard_action, lane, self.keys[lane], actions[lane])