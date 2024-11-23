import cv2
import os
import traceback 
import socket
import numpy as np
import threading
import time

def split_frames(in_path, out_path):
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

  def start(self):
    '''
    Starts the socket listener
    '''
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