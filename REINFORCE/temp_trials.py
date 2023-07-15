import gymnasium as gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ReinforceModel(nn.Module):
    """
    Define policy NN - Returns pi(a|s) = probability of each output action given the state
    """

    def __init__(self, num_action, num_input, action_space):
        super(ReinforceModel, self).__init__()
        self.num_action = num_action  # Number of output actions
        self.num_input = num_input  # Dimension of input state to model
        self.action_space = action_space  # Action space of the environment
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
        return np.random.choice(self.action_space,
                                p=action_prob.squeeze(0).detach().cpu().numpy())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
action_space = [0, 1]  # 0: Left, 1: Right
model = ReinforceModel(2, 4, action_space).to(device)

chkpt_dir = "./Models"
chkpt_file = os.path.join(chkpt_dir, 'Trained_Model')
model.load_state_dict(torch.load(chkpt_file))

env = gym.make("CartPole-v0", render_mode="human")
state = env.reset()[0]
for i in range(1000):
    env.render()
    action, log_prob = model(state)
    state, r_, done, i_, _ = env.step(action)
    if done:
        state = env.reset()[0]
    print(i)



          








    
