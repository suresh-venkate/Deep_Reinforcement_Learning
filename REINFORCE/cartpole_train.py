### Reference: https://github.com/kvsnoufal/reinforce

import gymnasium as gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReinforceModel(nn.Module):
    """
    Define policy NN - Returns pi(a|s) = probability of each output action given the state
    """
    def __init__(self, num_action, num_input, action_space):
        super(ReinforceModel, self).__init__()
        self.num_action = num_action  # Number of output actions
        self.num_input = num_input  # Dimension of input state to model
        self.action_space = action_space # Action space of the environment
        self.layer1 = nn.Linear(num_input, 64)  # Input Layer
        self.layer2 = nn.Linear(64, num_action)  # Output Layer
        
    def forward(self, x):
        """
        Forward pass through NN
        :param x: Input state
        :return:
        """
        # Convert input state vector to torch tensor.
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        x = F.relu(self.layer1(x))  # First layer with ReLu activation
        actions = F.softmax(self.layer2(x), dim=1)  # Obtain probability of output actions
        action = self.get_action(actions)  # Select action based on output probability
        log_prob_action = torch.log(actions.squeeze(0))[action]  # Log-probability of selected
                                                                 # action
        return action, log_prob_action
    def get_action(self, action_prob):
        """
        Pick an action based on the probability of each action
        :param action_prob: Action probability vector
        :return: Action picked at random based on the probability vector
        """
        return np.random.choice(self.action_space,\
                                p=action_prob.squeeze(0).detach().cpu().numpy())
    

num_episodes = 500  # Number of episodes to use for training
num_steps = 500  # Maximum number of steps to execute per episode
gamma = 0.9  # Discounting factor
render_flag = False  # If true, render environment

#env = gym.make("CartPole-v0", render_mode = "human")  # Instantiate environment
env = gym.make("CartPole-v0")  # Instantiate environment
action_space = [0, 1]  # 0: Left, 1: Right

model = ReinforceModel(2, 4, action_space).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
all_rewards = []

for episode in range(num_episodes):  # Loop through num_episodes
    done = False  # Reset 'done' flag
    state = env.reset()[0]  # Reset environment
    lp_array = []  # Placeholder to store array of log-prob of actions - Reset for every episode
    rew_array = []  # Placeholder to store reward received at each step - Reset for every episode
    disc_rew = []  # Placeholder to store discounted returns at each step

    # Execute episode
    for step in range(num_steps):
        if render_flag:  # Render environment if render_flag is 'True'
            env.render()
        # Forward-pass through NN to obtain (at) based on pi(a|s)
        # and log[pi(at|s)]
        action, log_prob = model(state)
        state, r_, done, i_, _ = env.step(action)  # Take env step based on current action
        lp_array.append(log_prob)  # Update lp_array with latest log_prob value (log[pi(at|s)])
        rew_array.append(r_)  # Update rew_array with current reward
        #print(step, done)
        if done:  # If end of episode is reached
            # Update all_rewards array with un-discounted
            # cumulative reward of current episode.
            all_rewards.append(np.sum(rew_array))
            if (episode % 100) >= 0:
                #print(f"EPISODE {episode} SCORE: {np.sum(r)} roll{pd.Series(all_rewards).tail(30).mean()}")
                print(f"Episode: {episode}; Score: {np.sum(rew_array)}")
            break
    # Episode completed

    # Episode post-processing
    # Compute cumulative discounted reward for each time step
    discounted_rewards = []  # Placeholder to store discounted_returns for each time step
    for t in range(len(rew_array)):  # Iterate through rew_array
        Gt = 0  # Initialize discounted return for current time-step
        pw = 0  # Initialize power 'pw' to 0
        # Compute discounted return for current time step
        for rew in rew_array[t:]:
            Gt = Gt + (gamma**pw) * rew
            pw = pw + 1
        discounted_rewards.append(Gt)  # Update discounted_rewards array

    # Perform learning step
    discounted_rewards = np.array(discounted_rewards)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
    discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards))/ (torch.std(discounted_rewards))
    log_prob = torch.stack(lp_array)
    policy_gradient = -log_prob * discounted_rewards  # Loss function
    model.zero_grad()  # Zero gradients
    policy_gradient.sum().backward()  # Back-prop
    optimizer.step()  # Update weights

chkpt_dir = "./Models"
chkpt_file = os.path.join(chkpt_dir, 'Trained_Model')
torch.save(model.state_dict(), chkpt_file)
