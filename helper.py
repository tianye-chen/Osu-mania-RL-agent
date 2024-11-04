import cv2
import os
import traceback 
import socket
import numpy as np

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
  def __init__(self, server='127.0.0.1', port=5555, data_handler=None):
    '''
    server: str: IP address of the server
    port: int: Port number to listen on
    data_handler: function: Function to handle incoming data
    
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
    self.data_handler = data_handler
    
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
    
  def start(self):
    '''
    Starts the socket listener
    '''
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
        print(f'Connection from {addr}')
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
      while True:
        data = conn.recv(4)
        if not data:
          break
      
        if self.data_handler:
          self.data_handler(data)
    except ConnectionResetError:
      print(f'Connection reset by {addr}')
    except Exception as e:
      print(e)
      traceback.print_exc()
    finally:
      conn.close()
      self.has_connection = False
      self.is_first_connection = False
      print(f'Connection closed.')
      
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
    