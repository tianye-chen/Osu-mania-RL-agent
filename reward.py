class SimulatedReward:
  def __init__(self, 
               begin_check=700,
               regular_note_threshold=[800, 930], 
               end_hold_threshold=[760, 890], 
               note_miss_threshold=[700, 960],
               custom_rewards=None
               ):
    '''
    Partial simulation of the game's reward system, only perfect and great hits are rewarded.
    
    keys_held should be handled separately in the main script
    
    begin_check: y threshold for notes to be considered for rewards
    regular_note_threshold: y threshold for normal and start holds to be rewarded
    end_hold_threshold: y threshold for end holds to be rewarded
    note_miss_threshold: y threshold for notes to be considered missed
    custom_rewards: [good_regular_notes, good_end_holds, broken_hold, bad_press, bad_release, missed_notes]
    
    '''
    self.hold_notes = [False] * 4
    self.keys_held = [False] * 4
    self.key_reference = ['s', 'd', 'k', 'l']
    
    self.begin_check = begin_check
    self.regular_note_threshold = regular_note_threshold
    self.end_hold_threshold = end_hold_threshold
    self.note_miss_threshold = note_miss_threshold
    
    self.debug_good_actions = [0, 0]
    self.debug_bad_actions = [0, 0, 0, 0]
    
    if custom_rewards:
      self.rewards = {
        'good_regular_notes': custom_rewards[0],
        'good_end_holds': custom_rewards[1],
        'broken_hold': custom_rewards[2],
        'bad_press': custom_rewards[3],
        'bad_release': custom_rewards[4],
        'missed_notes': custom_rewards[5]
      }
    else:
      self.rewards = {
        'good_regular_notes': 1,
        'good_end_holds': 1,
        'broken_hold': -1,
        'bad_press': -1,
        'bad_release': -1,
        'missed_notes': -5
      }
    
    
  def get_simulated_reward(self, keys, notes, verbose=False):
    '''
    
    keys: list of keys pressed, e.g. ['s', 'd', 'k', 'l']
    notes: list of notes in the current frame in [[class_id, lane, y_center], ...], can be any length
    '''
    if len(notes) == 0:
      return 0
    
    reward = 0
    notes = [x for x in notes if x[2] > self.begin_check][:4]

    # Check if hold notes are released early for each lane
    # hold_notes are handled by this function, 
    for i in range(len(self.hold_notes)):
      if self.hold_notes[i] and not self.keys_held[i]:
        self.debug_bad_actions[0] += 1
        self.hold_notes[i] = False
        reward += self.rewards['broken_hold']
        
        if verbose:
          print(f'Hold note broken')
    
    for note in notes:
      class_id, lane, y_center = note
      lane -= 1
      key_matched = any([self.key_reference.index(key) == lane for key in keys])
      
      # Check for normal (2) and start hold (3) notes
      if class_id in [2, 3]:
        if key_matched:
          if self.regular_note_threshold[0] < y_center <= self.regular_note_threshold[1]:
            reward += self.rewards['good_regular_notes']
            self.debug_good_actions[0] += 1
          elif not self.hold_notes[lane]:
            reward += self.rewards['bad_press']
            self.debug_bad_actions[1] += 1
            
            if verbose:
              print(f'Bad press for note at y {y_center}')
          
          # Update that a hold note is being held
          if class_id == 3:
            self.hold_notes[lane] = True

      # Check for end hold (1) notes
      if class_id == 1:
        if key_matched:
          if self.end_hold_threshold[0] < y_center <= self.end_hold_threshold[1]:
            reward += self.rewards['good_end_holds']
            self.debug_good_actions[1] += 1
          else:
            reward += self.rewards['bad_release']
            self.debug_bad_actions[2] += 1
            
            if verbose:
              print(f'Bad release for note at y {y_center}')
          
          # Update that a hold note is released   
          self.hold_notes[lane] = False
            
      # Check for missed notes
      if ((key_matched and not self.hold_notes[lane] and y_center < self.note_miss_threshold[0]) or 
          (not key_matched and y_center > self.note_miss_threshold[1])):
        reward += self.rewards['missed_notes']
        self.debug_bad_actions[3] += 1

        if verbose:
          if (key_matched and not self.hold_notes[lane]) and y_center < self.note_miss_threshold[0]:
            print(f'Missed note for too early at y {y_center}')
          elif not key_matched and y_center > self.note_miss_threshold[1]:
            print(f'Missed note for too late at y {y_center}')
              
    return reward
  
  def update_keys_held(self, keys_name: str, value: bool):
    '''
    Updates keys_held status
    
    keys_name: name of the key to be updated, in ['s', 'd', 'k', 'l']
    value: status of the key
    '''
    self.keys_held[self.key_reference.index(keys_name)] = value
    
  def get_debug(self):
    '''
    Returns a tuple of debug information
    
    Should only be used as a reference, not accurate when compared to the actual game
    
    returns (good_actions, bad_actions), a tuple of dictionaries
    '''
    good_actions = {
      'good_regular_notes': self.debug_good_actions[0],
      'good_end_holds': self.debug_good_actions[1]
    }
    
    bad_actions = {
      'broken_hold': self.debug_bad_actions[0],
      'bad_press': self.debug_bad_actions[1],
      'bad_release': self.debug_bad_actions[2],
      'missed_notes': self.debug_bad_actions[3]
    }
    
    return good_actions, bad_actions