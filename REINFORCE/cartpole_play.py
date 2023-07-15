import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

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
model.eval()

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255,255,255)
lineType = 2

fig = plt.figure()
env = gym.make("CartPole-v0", render_mode = "rgb_array")
state = env.reset()[0]
ims = []
rewards = []

for step in range(100):
    # env.render()
    img = env.render()
    # print(img)
    action, log_prob = model(state)
    #print(action)
    state, reward, done, _, _ = env.step(action)
    #print(reward)
    rewards.append(reward)
    #print(img.shape)
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    # Choose a font
    font = ImageFont.truetype("arial.ttf", 20)

    # Draw the text
    draw.text((0, 0), f"Step: {step} Action : {action} Reward: {reward} Total Rewards: {np.sum(rewards)} done: {done}", font=font,fill="#000000")

    # Save the image
    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    # img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2GRAY)
    im = plt.imshow(img, animated=True)
    ims.append([im])
    print(step)
env.close()

Writer = animation.writers['pillow']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                    blit=True)
im_ani.save('cp_train.gif', writer=writer)