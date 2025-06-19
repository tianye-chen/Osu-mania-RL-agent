import logging
import os
import pathlib
import subprocess
import argparse
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import mss
import numpy as np
import torch
from helper import SocketListener, capture, detect
from pynput.keyboard import Controller

KEY_PRESS = ['d', 'f', 'j', 'k']
Y_THRESHOLDS = [805, 860]  # Y thresholds for end_hold and press_note/start_hold notes

def press_key(actions, controller, verbose=False):
    # Presses and releases keys based on the action queue
    # class_ids reference:
    # 0: end_hold, 1: press_note, 2: start_hold
    
    if verbose:
        print(actions)
        
    for key, action_id in actions:
        if action_id in [1, 2]:
            controller.press(key)
        
    time.sleep(0.04)
    
    for key, action_id in actions:
        if action_id in [0, 1]:
            controller.release(key)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the osu!mania vision automation script.")
    parser.add_argument('-verbose', action='store_true', help="Enable verbose logging")
    args = parser.parse_args()
    
    if not os.path.exists('yolov5'):
        subprocess.Popen(["git", "clone", "https://github.com/ultralytics/yolov5"], stdout=subprocess.PIPE)
        subprocess.Popen(["pip", "install", "-r", "yolov5/requirements.txt"], stdout=subprocess.PIPE)
    if not os.path.exists('../osu'):
        raise FileNotFoundError("osu directory not found, make sure you have osu development build cloned into the parent directory")
        
    warnings.simplefilter("ignore", FutureWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    
    # Multithreading setup, required for simultaneous capture, detection, and action
    executor = ThreadPoolExecutor(max_workers=8)
    
    # Vision model setup
    try:
        vision_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt', force_reload=True)
    except Exception as e:
        pathlib.PosixPath = pathlib.WindowsPath # https://github.com/ultralytics/yolov5/issues/10240#issuecomment-1662573188
        vision_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt', force_reload=True)
    
    # Monitor setup for screen capture
    # !!! This was tested on a 1080p monitor, may not work on other resolutions
    monitor = mss.mss().monitors[1]
    t, l, w, h = monitor['top'], monitor['left'], monitor['width'], monitor['height']
    region = {'left': l+int(w * 0.338), 'top': t, 'width': w-int(w * 0.673), 'height': h} 
    
    # Socket listener setup, used to detect start and end of songs in-game
    listener = SocketListener()
    listener.start()
    song_begin = False
    
    # Keyboard controller setup, used to simulate key presses
    keyboard_controller = Controller()
    
    # Main automation loop
    while listener.is_first_connection or listener.is_listening:
        if listener.has_connection:
            song_begin = True
            action_queue = []
            
            # Captures the screen region and parses information into notes and locations
            image = capture(region)
            vision_thread = executor.submit(detect, np.array(image), vision_model)
            notes = vision_thread.result()
            
            # Act on the detected notes
            for note in notes:
                class_id, lane, y_center = note
                
                # Check for end_hold notes
                if y_center > Y_THRESHOLDS[0] and class_id == 0:
                    action_queue.append((KEY_PRESS[lane], class_id))
                # Check for press_notes and start_hold notes
                elif y_center > Y_THRESHOLDS[1]:
                    action_queue.append((KEY_PRESS[lane], class_id))
            
            if len(action_queue) > 0:
                executor.submit(press_key, action_queue, keyboard_controller, args.verbose)
                
        elif song_begin:
            listener.stop()
            break