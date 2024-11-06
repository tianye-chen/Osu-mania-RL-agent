import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pynput.keyboard import Controller
from concurrent.futures import ThreadPoolExecutor
from helper import SocketListener
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import mss
import pathlib

class OsuEnvironment(gym.Env):
    def __init__(self):
        # set frame per second
        self.frame_interval = 1 / 60
        # setup the neccessary resources for vision task
        self._vision_setup()
        # the keys corresponading to the 4 lane
        self.keys = ['s', 'd', 'k', 'l']

        # dyanamic list of vectors return tuple of (type, lane, y position) 
        # placehold for fixed size as 20
        self.observation = spaces.Box(low=0, high=np.inf, shape=(20, 3), dtype=np.float32)

        # 4 lane where each note can do nothing, pressed, held or release
        self.action_space = spaces.MultiDiscrete([4]*len(self.keys))
        
        # keep track of which key is hold
        self.currently_hold = [False] * len(self.keys)

        # check for invalid key press
        self.invalid = False
        self.keyboard = Controller()

        self.executor = ThreadPoolExecutor(max_workers=10)

        # socket setup
        self.data_queue = []
        self.hit_type = -1
        #self.listener = SocketListener(data_handler=self._data_handler)

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
    
        return ret 

    def _capture(self,region):
        with mss.mss() as sct:
            return sct.grab(region)
        
    # def _data_handler(self, data):
    #     self.data_queue.append(data)

    # def _check_connection(self):
    #     while True:
    #         if not self.listener.is_first_connection and not self.listener.has_connection:
    #             self.listener.stop()
    #             break
    
    # def socket_start(self):
    #     self.socket_thread = self.executor.submit(self._process_data_queue)
    #     self.executor.submit(self._check_connection)
    #     self.listener_thread = self.executor.submit(self.listener.start)

    # def socket_stop(self):
    #     self.listener.stop()

    def _notes_detection(self):
        vision_thread = self.executor.submit(self._capture, self.region)
        image = vision_thread.result()

        vision_thread = self.executor.submit(self._detect, np.array(image), self.model)
        self.observation = vision_thread.result()

    # def _process_data_queue(self):
    #     while self.listener.is_listening or self.listener.is_first_connection:
    #         if self.data_queue:
    #             data = self.data_queue.pop(0)
    #             data = int.from_bytes(data, byteorder='little')
    #             self.hit_type = data
            

    def reset(self):
        self.currently_hold = [False] * len(self.keys)
        self.observation = []
        self.data_queue = []
        self.invalid = False
        self.hit_type = -1

        return self.observation
    
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
                    time.sleep(0.05)
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
                
        # if there are invalid action
        if self.invalid:
            reward += -10
            self.invalid = False

        return reward

    def step(self, actions):
        time_start = time.time()
        # detect notes
        self._notes_detection()
        # take action based on the given actions simultaneously
        for lane in range(len(actions)):
            self.executor.submit(self._keyboard_action, lane, self.keys[lane], actions[lane])

        reward = 0
        truncate = False
        terminate = False
        if self.hit_type != -1:
            reward = self._get_reward(self.hit_type)
        
            # truncate and terminate based on hit type
            terminate = self.hit_type == 6
            truncate = self.hit_type == 7

            self.hit_type = -1
        
        # Frame rate control
        elapsed_time = time.time() - time_start
        if elapsed_time < self.frame_interval:
            time.sleep(self.frame_interval - elapsed_time)
        return self.observation, reward, truncate, terminate
