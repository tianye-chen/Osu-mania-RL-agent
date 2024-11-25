import cv2
import os
import traceback 
import socket
import numpy as np
import threading
import time
import mss

def split_frames(in_path, out_path):
  '''
  Splits video into frames, used for vision model training
  '''
  vid = cv2.VideoCapture(in_path)
  frame_idx = 0
  
  if not vid.isOpened():
    raise Exception(f"Error opening video file: {in_path}")
  
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  
  while True:
    ret, frame = vid.read()
    
    if not ret or frame is None:
      break
    
    frame_out = f"{out_path}/frame_{frame_idx:05d}.jpg"
    cv2.imwrite(frame_out, frame)
    
    frame_idx += 1
    
  vid.release()
  print(f"Wrote {frame_idx} frames to /{out_path}")
  
class DataQueue:
  '''
  Simple queue
  '''
  def __init__(self):
    self.queue = []
    
  def add(self, data):
    self.queue.append(data)
    
  def get(self):
    return self.queue
  
  def clear(self):
    self.queue = []

class SocketListener():
  def __init__(self, server='127.0.0.1', port=5555):
    '''
    server: str: IP address of the server
    port: int: Port number to listen on
  
    Raw data is passed to data_handler as bytes, make sure to decode
    
    Data is sent in integers and represents the following:
    
    0 - Miss \\
    1 - Meh \\
    2 - Ok \\
    3 - Good \\
    4 - Great \\
    5 - Perfect \\
    6 - Song passed \\
    7 - Song failed
    '''
    self.server = server
    self.port = port
    self.sock = None
    self.latest_data = None
    self.song_end = None
    self.data_handler = None

    #### Flags
    
    # Will be true at start, set to false after song ends or on stop()
    # Useful for checks before other flags are set due to timing issues
    self.is_first_connection = True  
    
    # True until _stop() is called 
    self.is_listening = False      
    
    # True if a song is playing   
    self.has_connection = False   
    
    # True if stop() is called    
    self.stop_requested = False       

    # Event flag for new data detection 
    self.has_new_data = threading.Event()

    # truncate when connection didn't close after song duration
    self.song_duration = 1000000

  def start(self, data_handler):
    '''
    Starts the socket listener
    '''
    self.data_handler = data_handler
    threading.Thread(target=self._listen, daemon=True).start()
    
  def _listen(self):
    try:
      self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.sock.bind((self.server, self.port))
      self.sock.listen(1)
      self.is_listening = True
      self.is_first_connection = True
      print(f'Listening on {self.server}:{self.port}')
      
      while True:
        if self.stop_requested:
          self._stop()
          break
        
        conn, addr = self.sock.accept()
        conn.settimeout(5)  # Timeout in seconds to avoid waiting forever
        #print(f'Connection from {addr}') disable print during tranining
        self.has_connection = True
        self._handle_connection(conn, addr)
    except Exception as e:
      print(e)
      traceback.print_exc()
      
  def _handle_connection(self, conn, addr):
    '''
    Internal function to handle incoming connections
    '''
    try:
      # reset when new song begin
      time_start = time.time()
      self.song_end = None
      self.latest_data = None
      while True:
        # calculate elapsed time
        time_elapsed = time.time() - time_start
        if time_elapsed > self.song_duration: # disconnect when over song duration
          self.song_end = 7
          break
        
        try:
          data = conn.recv(4)
          self.latest_data = int.from_bytes(data, byteorder='little')

          self.has_new_data.set() # signal that new data has arrived
          
          if self.data_handler:
            self.data_handler(self.latest_data)

          if self.latest_data in {6,7}:
            self.song_end = self.latest_data
            break

        except socket.timeout:
          continue

        if not data:
          self.song_end = 7
          break
    
    # disable print during tranining
    except ConnectionResetError:
      print(f'Connection reset by {addr}')
    except Exception as e:
      print(e)
      traceback.print_exc()
    finally:
      conn.close()
      self.has_connection = False
      self.is_first_connection = False
      # print(f'Connection closed.')
      
  def stop(self):
    self.stop_requested = True
      
  def _stop(self):
    if self.sock:
      self.sock.close()
      self.is_listening = False
      self.has_connection = False
      self.is_first_connection = False
      self.stop_requested = False
      print(f'Socket listener stopped.')
    else:
      print('Socket listener is not running.')
  
  def fetch_data(self, action_fuc, timeout):
    """
      Fetches data specifically after performing an action.
      action_func: function that triggers the keyboard action
      timeout: float: Time to wait for data, in seconds
    """
    # clear the event flag before performing action
    self.has_new_data.clear()

    # perform the action function
    action_fuc()

    # wait for new data within timeout
    if self.has_new_data.wait(timeout=timeout):
      return self.latest_data
    elif self.song_end is not None: # return when there is terminate or truncate signal
      return self.song_end
    else:
      return None # no data received
    
def preprocess_actions(actions):
    """
    Preprocess actions into shape [4,4,4,4]
    """
    key_hold = [False] * 4
    map_key = {"s": 0, "d" : 1, "k" : 2, "l" : 3}
    clean_actions = []
    for i in range(len(actions)):
        chars = actions[i]
        action = [0, 0, 0, 0]

        if chars == []:
            clean_actions.append(action)
            continue

        for char in chars:
            key = map_key.get(char)
            if key is None:
                continue
            
            if not key_hold[key]:
                action[key] = 1
            else:
                action[key] = 2

            if i + 1 < len(actions):
                if char not in actions[i+1]:
                    if key_hold[key]:
                        action[key] = 3            
                else:
                    key_hold[key] = True
                    action[key] = 2
                
        clean_actions.append(action)
    
    return clean_actions
  
def detect(img, model):
  '''
  Returns list of notes in the form of [class_id, lane, y_center] for a given image
  '''
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
    ret = sorted(ret, key=lambda note: (-note[2], note[1]))

  return ret 

def capture(region):
  with mss.mss() as sct:
    return sct.grab(region)
  
def pad_inner_array(arr, pad_value, pad_len):
  '''
  Pads inner arrays of a 2d array
  
  arr: 2d array
  pad_value: value to pad with
  pad_len: length to pad to
  
  '''
  padded = []
  
  for inner in arr:
    inner += [pad_value] * (pad_len - len(inner))
    padded.append(inner)
    
  return padded
