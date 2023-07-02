import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from util import plot_learning_curve

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        """
        :param lr: Learning rate for NN
        :param n_actions: Number of outputs required from NN
        :param input_dims: Dimension of input to NN
        """
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128) # Input layer
        self.fc2 = nn.Linear(128, n_actions) # Output layer

        # Define learning parameters
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss() # MSE loss used for training
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Run forward prop through the network
        :param state: Input to the NN to use for forward prop
        :return: Output of the NN
        """
        layer1 = F.relu(self.fc1(state)) # RELU activation for first layer
        actions = self.fc2(layer1)

        return actions


class Agent():
    """
    Defines agent class used in cartpole naive DQ learning project
    """
    def __init__(self, input_dims, n_actions, lr, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.lr = lr # Learning rate during training
        self.input_dims = input_dims # Shape of state/observation vector
        self.n_actions = n_actions # Number of outputs required from NN
        self.gamma = gamma # Discounting factor
        self.epsilon = epsilon # eps for eps-greedy action selection
        self.eps_dec = eps_dec # eps decay per step
        self.eps_min = eps_min # eps lower limit
        self.action_space = [i for i in range(self.n_actions)] # Action space of agent

        # Define agent network
        #   Input: State (S)
        #   Output is the Q value function of each action:
        #       Q(S,0): Left action-value function
        #       Q(S,q): Right action-value function
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        """
        epsilon-greedy action selection function
            - Exploitation: Select greedy action with probability (1 - eps)
            - Exploration: Select random action with probability (eps)
        :param observation: Current state
        :return: action to choose based on current state, current Q network and eps-greedy action selection
        """
        if np.random.random() > self.epsilon: # Exploitation with prob (1 - eps)
            state = T.tensor(observation, dtype=T.float).to(self.Q.device) # Convert state to tensor
            actions = self.Q.forward(state) # Obtain Q(S,a) for all actions
            action = T.argmax(actions).item() # Obtain greedy action
        else: # Exploration with prob (eps)
            action = np.random.choice(self.action_space) # Choose random action

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        """
        Q-learning function:
            - Q_curr = Q(St, At)
            - Q_target = R(t+1) + gamma * Q_max(S(t+1), A)
            - Update weights to minimize (Q_target - Q_curr) ** 2
        :param state: Current State (St)
        :param action: Current Action (At)
        :param reward: R(t+1)
        :param state_: Next State (S(t+1))
        :return: None
        """
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        # Estimate Q-target
        q_pred = self.Q.forward(states)[actions] # Q(St, At)
        q_next = self.Q.forward(states_).max()
        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device) # Estimate loss function
        loss.backward() # Obtain gradients
        self.Q.optimizer.step() # Back-prop
        self.decrement_epsilon() # Decrement epsilon

if __name__ == '__main__':
    # Instantiate Cart Pole gym environment
    # (https://gymnasium.farama.org/environments/classic_control/cart_pole/#cart-pole)
    # State: Each observation from environment is an array of shape (4,) with the following parameters:
    #   [0]: Cart Position
    #   [1]: Cart Velocity
    #   [2]: Pole Angle
    #   [3]: Pole Angular Velocity
    # Actions: The action space is a ndarray with shape(1, ) which can take values {0, 1}
    # indicating the direction of the fixed force the cart is pushed with.
    #   0: Push cart to the left
    #   1: Push cart to the right
    # Rewards: Reward of +1 obtained for every step taken including termination step
    # Starting State: All observations are assigned a uniformly random value in (-0.05, 0.05)
    # Episode end: The episode ends if any one of the following occurs:
    #   Termination: Pole Angle is greater than ±12°
    #   Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    #   Truncation: Episode length is greater than 500
    env = gym.make('CartPole-v1')
    #n_games = 5 # Number of episodes to train over
    n_games = 10000  # Number of episodes to train over
    scores = []
    eps_history = []

    # Define agent based on a FF NN
    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)

    for i in range(n_games): # Loop through n_games episodes
        score = 0 # Placeholder to track cumulative reward over the episode
        done = False # Episode done indicator flag
        obs = env.reset() # Reset observation and obtain initial state

        while not done: # Loop while episode is not done
            action = agent.choose_action(obs) # Choose action (At) based on eps-greedy policy
            obs_, reward, done, info = env.step(action) # Take environment step
            score += reward # Update score
            agent.learn(obs, action, reward, obs_) # Perform learning step
            obs = obs_ # S(t+1) = St
        scores.append(score) # Update current episode score in 'scores' list
        eps_history.append(agent.epsilon) # Update eps_history with epsilon at end of episode

        if i % 100 == 0: # Print some learning statistics every 100 episodes
            avg_score = np.mean(scores[-100:])
            print('Episode: %d, Score: %0.1f. Avg Score: %0.1f, Epsilon: %0.2f' %
                  (i, score, avg_score, agent.epsilon))

    # Plot learning curve to a file
    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)

    ### Summary: This method achieves poor learning as expected since we are using a naive NN.
    ### DQN will help to improve the performance