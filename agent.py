from utils import DQN, ReplayMemory

class Agent:
    def __init__(self, 
                 env, 
                 criterion, 
                 optimizer, 
                 discount_factor=0.99, 
                 epsilon_decay=100, 
                 epsilon_start=0.95,
                 epsilon_end=0.10, 
                 target_update_rate=0.005, 
                 batch_size=128, 
                 capacity=10000,
                 dueling_dqn=False,
                 noisy_dqn=False,
                 behavior_cloning=False,
                 std_init=0.5):
        
        #define the environment
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # define hyperparameter
        self.criterion = criterion
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.capacity = capacity
        self.noisy_dqn = noisy_dqn
        self.epsilon_update = 0

        # define the dqn
        input_size = sum(self.observation_space.shape)
        output_size = self.action_space.n
        self.policy_net = DQN(input_size, output_size, dueling=dueling_dqn, noisy=noisy_dqn, std_init=std_init)
        self.target_net = DQN(input_size, output_size, dueling=dueling_dqn, noisy=noisy_dqn, std_init=std_init)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.experience_replay = ReplayMemory(self.capacity)
        self.expert_replay = None # store expert replay for demonstration

        # track reward and loss
        self.rewards = []
        self.loss = []

    def action_policy(self, state):
        if self.noisy_dqn:
            
