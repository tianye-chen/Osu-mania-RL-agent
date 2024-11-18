import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pynput.keyboard import Controller, Key
from concurrent.futures import ThreadPoolExecutor
from helper import SocketListener
import torch
import mss
import pathlib
from collections import deque
import time

class OsuEnvironment(gym.Env):
    def __init__(self):
        # set frame per second
        self.frame_interval = 1 / 15

        # setup the neccessary resources for vision task
        self._vision_setup()

        # the keys corresponading to the 4 lane
        self.keys = ['s', 'd', 'k', 'l']

        # stacked frames of notes
        self.observation = deque(maxlen=4)

        # returns 4 stacked of list of note vectors in [type, lane, y position]
        # maximum notes return
        self.max_notes = 8
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
        self.listener.start()

        # remeber the song and mode it select
        self.song = ""
        self.mode = 0
        self.duration = 0

        # song bank for randomize 
        self.song_dict = {
            "#be_fortunate": [4, 2*60+14],
            "akkera-country-boy": [7, 1*60+55],
            "aresene's bazaar": [1, 2*60+25],
            "b.b.k.k.b.k.k. (hurce remix)": [1, 4*60+19],
            "bassline yatteru? w": [1, 4*60+35],
            "break the silence": [1, 4*60+24],
            "burst the gravity (tv size)": [4, 1*60+29],
            "candy luv (short ver.)": [2, 2*60+12],
            "cyberia lyr 3": [10, 3*60+31],
            "empire": [4, 2*60+43],
            "enchanted love": [3, 2*60+8],
            "eutopia (tv size)": [4, 1*60+31],
            "flaklypa": [1, 2*60+59],
            "free at last": [1, 2*60+32],
            "ghoul": [8, 4*60+16],
            "heart function": [4, 2*60+25],
            "hype feat. such (sky_delta remix)": [1, 3*60+56],
            "iced tea for breakfast": [3, 1*60+4],
            "innocent letter": [1, 3*60+46],
            "koiyamai (tv size)": [5, 1*60+39],
            "liquated": [1, 4*60+20],
            "liquid (paul rosenthal remix)": [1, 2*60+38],
            "million pp": [1, 7*60+5],
            "machinegun poem doll": [6, 2*60+15],
            "memoria reborn": [1, 3*60+43],
            "mopemope": [3, 1*60+47],
            "more! jump! more!": [5, 2*60],
            "mutsuki akari no yuki": [1, 2*60+21],
            "my love": [2, 3*60+42],
            "never give up": [1, 2*60+23],
            "new world": [4, 2*60+2],
            "otome no route wa hitotsu ja nai! (tv size)": [4, 1*60+26],
            "palette gamma": [8, 1*60+23],
            "regret": [5, 1*60+58],
            "renatus": [1, 3*60+28],
            "rpg": [1, 6*60+1],
            "run*2 run to you!!": [5, 1*60+53],
            "specialist (cut ver.)": [5, 2*60+35],
            "suki yo ~two hearts~": [4, 3*60+30],
            "tear rain": [1, 4*60+1],
            "teo": [4, 3*60+22],
            "the empress": [4, 4*60+5],
            "the light": [1, 1*60+46],
            "time files": [2, 1*60+38],
            "triangles": [1, 2*60+1],
            "tsuioku no bara": [1, 5*60+26],
            "victorious journey": [1, 3*60+54],
            "wave feat. aitsuki nakuru": [1, 3*60+28],
            "we could get more machinegun psystyle! (and more genre switches)": [5, 5*60+15],
            "yuriyurarararayuruyuri daijiken (tv size)": [4, 1*60+26]
        }   

    def reset(self):
        # time for switching to the game and connection to reset
        time.sleep(10)
        self.currently_hold = [False] * len(self.keys)
        self.invalid = False
        self.observation.clear()
        note_vectors = []
        for _ in range(self.max_notes):
            note_vectors.append([0,0,0])

        for _ in range(self.observation.maxlen):
            self.observation.append(note_vectors)

        return self.observation
    
    def checking_connection(self):
        return self.listener.is_listening or self.listener.is_first_connection
    
    def song_begin(self):
        return self.listener.has_connection
    
    def getSong(self):
        return f"{self.song}, mode: {self.mode}"
    
    def step(self, actions):
        time_start = time.time()
        reward = 0
        truncate = False
        terminate = False

        # detect notes
        self._notes_detection()

        # take action based on the given actions simultaneously and fetech the hit type after action
        data = self.listener.fetch_data(action_fuc=lambda: self._perform_action(actions), timeout=0.02)

        if data is not None:
            reward, truncate, terminate = self._get_reward(data)

        # ensure that each step represents proper FS frame
        time_end = time.time() - time_start

        if time_end < self.frame_interval:
            time.sleep(self.frame_interval-time_end)

        return self.observation, reward, truncate, terminate

    def lost_connection(self):
        return not self.listener.has_connection
        
    def pick_random_song(self):
        # empty search bar
        self.executor.submit(self._key_press, 'a').result()
        self.executor.submit(self._key_press, Key.esc).result()

        # get random song
        song_index = np.random.randint(0, len(self.song_dict))
        self.song = list(self.song_dict.keys())[song_index]

        # enter the song name in the search bar
        for char in self.song:
            self.executor.submit(self._key_press, char).result()

        time.sleep(1) # time for the search to showup

        self.mode, self.duration = self.song_dict.get(self.song)

        if self.mode != 1:
            self.mode = np.random.randint(1, self.mode+1)
            if self.mode != 1:
                for _ in range(self.mode-1):
                    self.executor.submit(self._key_press, Key.down).result()

        self.executor.submit(self._key_press, Key.enter).result()
        self.listener.song_duration = self.duration + 30
        time.sleep(3) # for socket to connect

        self.executor.submit(self._skip_cutscene, 5)

    def return_to_song_selection_after_song(self):
        time.sleep(5) # wait for the fail/success ui to popup
        self.executor.submit(self._key_press, Key.esc).result()

        # clear potential character in search bar resulted from step 
        self.executor.submit(self._key_press, 'a').result()
        self.executor.submit(self._key_press, Key.esc).result()

        # get the the song selection back to the first mode
        if self.mode > 1:
            for char in self.song:
                self.executor.submit(self._key_press, char).result()
            time.sleep(1)
            for _ in range(self.mode-1):
                self.executor.submit(self._key_press, Key.up).result()

            self.executor.submit(self._key_press, Key.esc).result()

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
            case 7: # failure
                truncate = True
                
        # if there are invalid action
        if self.invalid:
            reward += -10
            self.invalid = False

        return reward, truncate, terminate

    def _vision_setup(self):
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

        if ret:
            # only care about notes that are near the hit window
            ret = sorted(ret, key=lambda note: (-note[2], note[1]))
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
        self.observation.append(vision_thread.result())

    def _perform_action(self, actions):
        if isinstance(actions, torch.Tensor): 
            actions = actions.tolist()
        if self.song_begin:
            keys = []
            for lane in range(len(actions)):
                key = self.executor.submit(self._keyboard_action, lane, self.keys[lane], actions[lane])
                keys.append(key)

            for key in keys:
                key.result()
    
    def _key_press(self, key):
        self.keyboard.press(key)
        time.sleep(0.2)
        self.keyboard.release(key)

    def _skip_cutscene(self, sec):
        time.sleep(sec)
        self._key_press(Key.space)