import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    """
    Class to define the NN to use for the model
    """
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir  # Directory to store checkpoint files
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)  # Name of checkpoint file

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)  # Input convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # Second convolutional layer
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)  # Final convolutional layer

        # Compute dimension of the output of the final convolutional layer
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)  # First FC layer
        self.fc2 = nn.Linear(512, n_actions)  # Output FC layer

        # Define RMS-prop optimizer and MSE loss
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    def calculate_conv_output_dims(self, input_dims):
        """
        Function to calculate the output dimension of the third convolutional layer
        in the deep-Q network
        :param input_dims:  Dimensions of the input to the first convolutional layer
        :return: Dimension of the output of the third convolutional layer
        """
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        """
        Forward-Prop function
        :param state: Input state (st)
        :return: Q(st, a)
        """
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)

        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        """
        Function to save checkpoint
        :return: None
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Function to load checkpoint
        :return: None
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))