class Reward:
  def __init__(self, 
               begin_check=700,
               regular_note_threshold=[800, 930], 
               end_hold_threshold=[760, 890], 
               note_miss_threshold=[700, 960],
               ):
    '''
    Partial simulation of the game's reward system, only perfect and great hits are rewarded.
    
    keys_held should be handled separately in the main script through update_keys_held
    
    begin_check: y threshold for notes to be considered for rewards
    regular_note_threshold: y threshold for normal and start holds to be rewarded
    end_hold_threshold: y threshold for end holds to be rewarded
    note_miss_threshold: y threshold for notes to be considered missed    
    '''
    self.hold_notes = [False] * 4
    self.keys_held = [False] * 4
    self.key_reference = ['s', 'd', 'k', 'l']
    
    self.begin_check = begin_check
    self.regular_note_threshold = regular_note_threshold
    self.end_hold_threshold = end_hold_threshold
    self.note_miss_threshold = note_miss_threshold
    
    self.debug_good_actions = [0, 0, 0]
    self.debug_bad_actions = [0, 0, 0, 0, 0, 0]
    self.debug_in_game_actions = [0, 0, 0, 0, 0, 0, 0, 0]
    
    self.rewards = {
      'good_regular_notes': 6,
      'good_end_holds': 6,
      'good_hold': 1,
      'bad_hold': -1,
      'broken_hold': -3,
      'bad_press': -2,
      'bad_release': -2,
      'missed_notes': -6,
      'unnecessary_press': -1
    }
    
  def get_simulated_reward(self, keys, notes, verbose=False):
    '''
    Returns reward for given frame with respect to the notes and keys pressed
    
    keys: list of keys pressed, e.g. [0, 1, 2, 3]
    notes: list of notes in the current frame in [[class_id, lane, y_center], ...], can be any length
    '''
    reward = 0
    if len(notes) == 0:
      if len(keys) > 0:
        reward += self.rewards['unnecessary_press']
        self.debug_bad_actions[5] += 1
        
        if verbose:
          print(f'Unnecessary press')
      
      return reward
    
    notes = [x for x in notes if x[2] > self.begin_check][:4]

    # Check if hold notes are released early for each lane
    # hold_notes are handled by this function, keys_held should be handled separately
    for i in range(len(self.hold_notes)):
      if self.hold_notes[i] and not self.keys_held[i]:
        self.debug_bad_actions[1] += 1
        self.hold_notes[i] = False
        reward += self.rewards['broken_hold']
        
        if verbose:
          print(f'Hold note broken')
      elif not self.hold_notes[i] and self.keys_held[i]:
        self.debug_bad_actions[0] += 1
        reward += self.rewards['bad_hold']
      elif self.hold_notes[i] and self.keys_held[i]:
        self.debug_good_actions[2] += 1
        reward += self.rewards['good_hold']
    
    for note in notes:
      class_id, lane, y_center = note
      lane -= 1
      key_matched = any([key == lane for key in keys])
      
      # Check if there exists a key press for a corresponding note in any lane
      if key_matched:
        # Check for normal (2) and start hold (3) notes
        if class_id in [2, 3]:
          if self.regular_note_threshold[0] < y_center <= self.regular_note_threshold[1]:
            reward += self.rewards['good_regular_notes']
            self.debug_good_actions[0] += 1
          elif not self.hold_notes[lane]:
            reward += self.rewards['bad_press']
            self.debug_bad_actions[2] += 1
            
            if verbose:
              print(f'Bad press for note at y {y_center}')
          
          # Update that a hold note is being held
          if class_id == 3:
            self.hold_notes[lane] = True

        # Check for end hold (1) notes
        if class_id == 1:
          if self.end_hold_threshold[0] < y_center <= self.end_hold_threshold[1]:
            reward += self.rewards['good_end_holds']
            self.debug_good_actions[1] += 1
          else:
            reward += self.rewards['bad_release']
            self.debug_bad_actions[3] += 1
            
            if verbose:
              print(f'Bad release for note at y {y_center}')
          
          # Update that a hold note is released   
          self.hold_notes[lane] = False
              
      # Check for missed notes
      if ((key_matched and not self.hold_notes[lane] and y_center < self.note_miss_threshold[0]) or 
          (not key_matched and y_center > self.note_miss_threshold[1])):
        reward += self.rewards['missed_notes']
        self.debug_bad_actions[4] += 1

        if verbose:
          print(f'Missed note at y {y_center}')
              
    return reward
  
  def update_keys_held(self, keys_idx, value: bool):
    '''
    Updates keys_held status
    
    keys_idx: idx of the key to be updated, in [0, 1, 2, 3]
    value: status of the key
    '''
    self.keys_held[keys_idx] = value
    
  def get_key_held(self, key_idx):
    '''
    Returns the key_held status of a key
    
    key_idx: idx of the key to be checked, in [0, 1, 2, 3]
    '''
    return self.keys_held[key_idx]
  
  def set_custom_rewards(self, custom_rewards):
    '''
    Override default rewards with custom rewards
    
    custom_rewards: [good_regular_notes, 
                    good_end_holds, 
                    good_hold,
                    bad_hold,
                    broken_hold, 
                    bad_press, 
                    bad_release,
                    missed_notes,
                    unnecessary_press]
    '''
    if len(custom_rewards) != 9:
      raise ValueError('Custom rewards should be of length 9')
    
    self.rewards = {
      'good_regular_notes': custom_rewards[0],
      'good_end_holds': custom_rewards[1],
      'good_hold': custom_rewards[2],
      'bad_hold': custom_rewards[3],
      'broken_hold': custom_rewards[4],
      'bad_press': custom_rewards[5],
      'bad_release': custom_rewards[6],
      'missed_notes': custom_rewards[7],
      'unnecessary_press': custom_rewards[8]
    }
      
    
  def get_debug(self, render_mode=False):
    '''
    Returns a tuple of debug information
    
    Should only be used as a reference, not always accurate when compared to the actual game
    Resets debug counters being called
    
    returns in_game_actions or (good_actions, bad_actions), depending on render_mode
    '''
    if render_mode:
      in_game_actions = {
        'miss': self.debug_in_game_actions[0],
        'bad': self.debug_in_game_actions[1],
        'meh': self.debug_in_game_actions[2],
        'ok': self.debug_in_game_actions[3],
        'great': self.debug_in_game_actions[4],
        'perfect': self.debug_in_game_actions[5],
        'song_cleared': self.debug_in_game_actions[6],
        'song_failed': self.debug_in_game_actions[7]
      }
      
      self.debug_in_game_actions = [0, 0, 0, 0, 0, 0, 0, 0]
      
      return in_game_actions
    else:
      good_actions = {
        'good_regular_notes': self.debug_good_actions[0],
        'good_end_holds': self.debug_good_actions[1],
        'good_hold': self.debug_good_actions[2]
      }
      
      bad_actions = {
        'bad_hold': self.debug_bad_actions[0],
        'broken_hold': self.debug_bad_actions[1],
        'bad_press': self.debug_bad_actions[2],
        'bad_release': self.debug_bad_actions[3],
        'missed_notes': self.debug_bad_actions[4],
        'unnecessary_press': self.debug_bad_actions[5]
      }
      
      self.debug_good_actions = [0, 0, 0]
      self.debug_bad_actions = [0, 0, 0, 0, 0, 0]
      
      return good_actions, bad_actions
  
  
  def get_in_game_reward(self, hit_actions):
    '''
    Returns rewards for hit action results provided by the game
    '''
    reward = 0

    for i in hit_actions:
      self.debug_in_game_actions[i] += 1
      if i == 0:
        reward += -1
      elif i == 1:
        reward += -0.2
      elif i == 2:
        reward += 0.5
      elif i == 3:
        reward += 1
      elif i == 5:
        reward += 1.5
      elif i == 6:
        reward -= 10
      elif i == 7:
        reward += 0
          
    return reward
        
        