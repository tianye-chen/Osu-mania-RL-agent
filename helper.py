import cv2
import os

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