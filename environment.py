import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pynput.keyboard import Controller, Key
from concurrent.futures import ThreadPoolExecutor
from helper import SocketListener,  DataQueue, capture, detect
import torch
import mss
import pathlib
from collections import deque
import time

class OsuEnvironment(gym.Env):
    def __init__(self, num_frame = 4, max_notes = 8, monitor_id = 1, conn_print=False):
        # setup the neccessary resources for vision task
        self.monitor_id = monitor_id
        self._vision_setup()

        # the keys corresponading to the 4 lane
        self.keys = ['s', 'd', 'k', 'l']

        # stacked frames of notes
        self.observation = deque(maxlen=num_frame)

        # returns n stacked of list of note vectors in [type, lane, y position]
        # maximum notes return
        self.num_frame = num_frame
        self.max_notes = max_notes
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_frame, self.max_notes, 3), dtype=np.float32)

        # 4 lane where each note can do nothing, pressed, held or release
        self.action_space = spaces.MultiDiscrete([4]*len(self.keys))
        
        # keep track of which key is hold and if key is pressed
        self.currently_hold = [False] * len(self.keys)

        # check for invalid key press
        self.invalid = False
        self.keyboard = Controller()
        
        # create a number of thread for use
        self.executor = ThreadPoolExecutor(max_workers=7)

        # socket setup
        self.listener = SocketListener()
        self.hit_data = DataQueue()
        self.listener.start(data_handler=self.hit_data.add, show_print=conn_print)

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
            "da xi (sped up ver.)": [2, 2*60+18],
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

        # song use for evaluate
        self.test_song_dict = {
            "aresene's bazaar": [1, 2*60+25],
            "burst the gravity (tv size)": [1, 1*60+29],
            "candy luv (short ver.)": [1, 2*60+12],
            "da xi (sped up ver.)": [2, 2*60+18],
            "empire": [2, 2*60+43],
            "enchanted love": [2, 2*60+8],
            "eutopia":[4, 1*60+31],
            "heart function":[3, 2*60+25],
            "teo": [4, 3*60+22],
            "ghoul": [2, 4*60+4]
        }

    def reset(self):
        # time for switching to the game and connection to reset
        time.sleep(10)
        self.currently_hold = [False] * len(self.keys)
        self.invalid = False
        self.observation.clear()
        self.hit_data.clear()
        note_vectors = [[0,0,0]] * self.max_notes
        for _ in range(self.num_frame):
            self.observation.append(note_vectors)

        return self.observation
    
    def checking_connection(self):
        return self.listener.is_listening or self.listener.is_first_connection
    
    def song_begin(self):
        return self.listener.has_connection 
    
    def getSong(self):
        return f"{self.song}, mode: {self.mode}"
    
    def step(self, actions, train=True):
        reward = 0
        truncate = False
        terminate = False

        # detect notes
        self._notes_detection()

        # take action based on the given actions simultaneously and fetech the hit type after action
        self.hit_data.clear()
        self._perform_action(actions)
        data = self.hit_data.get()
        info = {"idle": True} # it means that there is no hit type recieved

        if len(data) != 0:
            reward, truncate, terminate = self._get_reward(data)
            info["idle"] = False

        # check in case it isn't in the data queue
        if self.listener.song_end == 6:
            terminate = True

        if self.listener.song_end == 7:
            truncate = True
            
        return self.observation, reward, truncate, terminate, info

    def lost_connection(self):
        return not self.listener.has_connection
        
    def pick_random_song(self, training=True, index=0):
        # empty search bar
        self.executor.submit(self._key_press, 'a').result()
        self.executor.submit(self._key_press, Key.esc).result()

        # get random song
        song_index = np.random.randint(0, len(self.song_dict))
        self.song = list(self.song_dict.keys())[song_index] if training else list(self.test_song_dict.keys())[index]

        # enter the song name in the search bar
        for char in self.song:
            self.executor.submit(self._key_press, char).result()

        time.sleep(1) # time for the search to showup

        self.mode, self.duration = self.song_dict.get(self.song) if training else self.test_song_dict.get(self.song)

        if self.mode != 1:
            self.mode = np.random.randint(1, self.mode+1) if training else self.mode
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
            case 2: # press
                if self.currently_hold[lane]:
                    self.invalid = True
                else:
                    self.keyboard.press(key)
                    self.keyboard.release(key)
            case 3: # hold
                if self.currently_hold[lane]:
                    return
                self.keyboard.press(key)
                self.currently_hold[lane] = True
            case 1: # release
                if self.currently_hold[lane]:
                    self.keyboard.release(key)
                    self.currently_hold[lane] = False
                else:
                    self.invalid = True

    def _get_reward(self, data):
        reward = 0 
        truncate = False
        terminate = False
        # reward based on the action taken
        for hit_type in data:
            match hit_type:
                case 0: # miss
                    reward += -3
                case 1: # meh
                    reward += -2
                case 2: # ok
                    reward += -1
                case 3: # good
                    reward += 1
                case 4: # great
                    reward += 4
                case 5: # perfect
                    reward += 6
                case 6: # pass
                    terminate = True
                case 7: # failure
                    truncate = True
                
        # if there are invalid action
        if self.invalid:
            reward += -5
            self.invalid = False

        return reward, truncate, terminate

    def _vision_setup(self):
        pathlib.PosixPath = pathlib.WindowsPath # https://github.com/ultralytics/yolov5/issues/10240#issuecomment-1662573188
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt', force_reload=True)  

        self.monitor = mss.mss().monitors[self.monitor_id]
        t, l, w, h = self.monitor['top'], self.monitor['left'], self.monitor['width'], self.monitor['height']
        self.region = {'left': l+int(w * 0.338), 'top': t, 'width': w-int(w * 0.673), 'height': h} 
        
    def _notes_detection(self):
        vision_thread = self.executor.submit(capture, self.region)
        image = vision_thread.result()

        vision_thread = self.executor.submit(detect, np.array(image), self.model, True)
        note_vector = vision_thread.result()

        note_vector = note_vector[:self.max_notes]
        note_vector += [[0,0,0]] * (self.max_notes - len(note_vector))

        self.observation.append(note_vector)

    def _perform_action(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.squeeze(0)
            actions = actions.cpu().tolist()

        if self.song_begin:
            threads = []
            for lane in range(len(actions)):
                key_thread = self.executor.submit(self._keyboard_action, lane, self.keys[lane], actions[lane])
                threads.append(key_thread)

            for key in threads:
                key.result()
        
        time.sleep(0.01)
    
    def _key_press(self, key):
        self.keyboard.press(key)
        time.sleep(0.2)
        self.keyboard.release(key)

    def _skip_cutscene(self, sec):
        time.sleep(sec)
        self._key_press(Key.space)