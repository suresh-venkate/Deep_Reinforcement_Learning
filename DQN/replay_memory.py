import numpy as np

class ReplayBuffer(object):
    """
    Replay Memory buffer class
    Store multiple samples of [st, at, rt, s(t+1)]
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size  # Maximum replay memory buffer size
        self.mem_cntr = 0  # Initialize memory location counter
        self.state_memory = np.zeros((self.mem_size, *input_shape),\
                                     dtype=np.float32)  # Buffer for 'st'
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),\
                                         dtype=np.float32)  # Buffer for 's(t+1)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)  # Buffer for at
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)  # Buffer for rt
        #self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)  # Buffer for 'done' flag
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)  # Buffer for 'done' flag

    def store_transition(self, state, action, reward, state_, done):
        """
        Function to store current datapoint in replay memory
        :param state: Current state (st)
        :param action: Current action (at)
        :param reward: Current reward (rt)
        :param state_: Next state [s(t+1)]
        :param done: 'Done' flag
        :return: None
        """
        index = self.mem_cntr % self.mem_size  # Current memory index pointer
        self.state_memory[index] = state  # Store current state (st)
        self.new_state_memory[index] = state_  # Store next state [s(t+1)]
        self.action_memory[index] = action  # Store current action (at)
        self.reward_memory[index] = reward  # Store current reward (rt)
        self.terminal_memory[index] = done  # Store 'done' flag
        self.mem_cntr += 1  # Increment memory counter

    def sample_buffer(self, batch_size):
        """
        Function to sample a buffer of size 'batch_size' from replay memory
        :param batch_size: Size of batch to sample
        :return: Sampled batch (states, actions, rewards, state_, terminal)
        """

        max_mem = min(self.mem_cntr, self.mem_size)  # Sample only from replay memory locations
                                                     # that have a valid stored sample
        batch = np.random.choice(max_mem, batch_size, replace=False)  # Sampled batch (sampling w/o
                                                                      # replacement)
        states = self.state_memory[batch]  # Extract current-states array ('st' array) from sampled batch
        actions = self.action_memory[batch]  # Extract actions array ('at' array) from sampled batch
        rewards = self.reward_memory[batch]  # Extract rewards array ('rt' array) from sampled batch
        states_ = self.new_state_memory[batch]  # Extract next-states array ('s(t+1) array) from sampled batch
        terminal = self.terminal_memory[batch]  # Extract 'done-state' array from sampled batch

        return states, actions, rewards, states_, terminal
