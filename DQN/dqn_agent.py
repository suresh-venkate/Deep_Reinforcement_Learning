import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma  # Discounting factor
        self.epsilon = epsilon  # epsilon used for eps-greedy action selection
        self.lr = lr  # Learning rate to use while training
        self.n_actions = n_actions  # Number of actions supported by the environment
        self.input_dims = input_dims  # Shape of the state input to the NN
        self.batch_size = batch_size  # Batch size to use while training
        self.eps_min = eps_min  # Lower clamp for epsilon
        self.eps_dec = eps_dec  # Rate at which epsilon is decreased
        self.replace_target_cnt = replace  # Update target network weights every 'replace_target_cnt' steps
        self.algo = algo  # Name of algorithm - Used in the checkpoint file name
        self.env_name = env_name  # Name of the environment - Used in the checkpoint file name
        self.chkpt_dir = chkpt_dir  # Directory to store the checkpoint files
        self.action_space = [i for i in range(n_actions)]  # Define action space
        self.learn_step_counter = 0  # Counter to keep track of every env (or learning) step

        # Instantiate Replay Memory buffer
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        # Instantiate Main DQ NN - Used for action selection as well
        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name+'_'+self.algo+'_q_eval',
                                   chkpt_dir=self.chkpt_dir)
        # Instantiate target DQ NN
        self.q_next = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        """
        Function to perform eps-greedy action selection
        :param observation:  Input state
        :return:  Output action
        """
        if np.random.random() > self.epsilon:  # Exploitation: Pick greedy action with prob: (1 - eps)
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:  # Exploration: Pick a random action with prob: eps
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        """
        Function to store one sample in replay memory
        :param state: Current state (st)
        :param action: Current action (at)
        :param reward: Current reward (rt)
        :param state_: Next state [s(t+1)]
        :param done: 'Done' flag
        :return: None
        """
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        """
        Sample a batch from replay memory
        :return: Sampled values
        """
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        # Update target network weights for every 'replace_target_cnt' steps
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        """
        Function to decrement epsilon after every env. step
        :return: None
        """
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        """
        Function to save checkpoints of eval and target networks
        :return: None
        """
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        """
        Function to load saved checkpoints for eval and target networks
        :return: None
        """
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        """
        Function to perform learning step
        :return:None
        """
        # Wait until replay memory has at least one batch of real samples
        # before performing learning step
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()  # Zero out gradients
        self.replace_target_network()  # Update target network weights

        # Sample one batch from replay memory
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        # Obtain q_pred from eval network
        q_pred = self.q_eval.forward(states)[indices, actions]

        # Obtain q_target from target network
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        # Learning step
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)  # Obtain loss function
        loss.backward()  # Back-prop
        self.q_eval.optimizer.step()  # Update weights
        self.learn_step_counter += 1  # Increment learn_step_counter
        self.decrement_epsilon()  # Decrement epsilon