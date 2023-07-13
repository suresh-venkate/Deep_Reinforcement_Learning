import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class RepeatActionAndMaxFrame(gym.Wrapper):
    """
    Class to repeat action for four frames and take maximum of
    the last two frames
    """
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat # Number of frames for which current action will be repeated
        self.shape = env.observation_space.low.shape # Shape of each output frame
        #self.frame_buffer = np.zeros_like((2, self.shape)) # Buffer to hold two frames
        self.frame_buffer = np.zeros_like((1, 1))  # Buffer to hold two frames
        self.clip_reward = clip_reward # Clip_reward flag
        self.no_ops = no_ops  # Number of steps for which no operation will be taken during
                              # initialization. Used only during testing and not during training.
        self.fire_first = fire_first  # Used to fire first after executing no_ops.
                                      # Used only during testing and not during training

    def step(self, action):
        """
        Step function for wrapped environment
            1) Current action repeated for four frames

        :param action: Action to use for current step
        :return: (max_frame, t_reward, done, info), where:
                    max_frame: Maximum of each pixel over the last two frames
                    t_reward: Total reward over the last 4 frames
        """
        t_reward = 0.0 # Initialize total reward (over 4 frames) to 0
        done = False # Initialize 'done' flag to False
        for i in range(self.repeat): # Repeat for 4 frames
            obs, reward, done, info, _ = self.env.step(action) # Take env step
            if self.clip_reward:  # Clip reward if applicable
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward # Increment total reward with reward from current step
            idx = i % 2
            self.frame_buffer[idx] = obs # Update frame buffer with last 2 frames.
                                         # First two frames out of the 4 frames are dropped
            if done: # Break out of loop if terminate state reached
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1]) # Obtain max. of previous 2 frames
                                                                           # pixel by pixel
        return max_frame, t_reward, done, info

    def reset(self):
        """
        Reset wrapped environment
        :return: Reset state
        """
        obs = self.env.reset() # Reset base environment and obtain observation
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0  # Not clear
        for _ in range(no_ops):  # Perform no_ops at beginning if flag is set
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:  # Fire first if flag is set
            # Check whether action '1' corresponds to 'FIRE'
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            # Execute action 'FIRE'
            obs, _, _, _ = self.env.step(1)

        # Initialize frame buffer
        self.frame_buffer = np.zeros((2, obs[0].shape[0], obs[0].shape[1], obs[0].shape[2]))
        self.frame_buffer[0] = obs[0]  # Update frame buffer with obs.
        #self.frame_buffer[0] = obs  # Update frame buffer with obs.

        #return obs
        return obs[0]

class PreprocessFrame(gym.ObservationWrapper):
    """
    Wrapper to pre-process the frames
        - Convert to gray scale
        - Resize to new shape
        - Swap channels axis to be compatible with Pytorch notation
        - Scale to [0.0, 1.0]
    """
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1]) # Swap channels axis
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=self.shape, dtype=np.float32) # Update observation space to
                                                                        # new shape and new scale

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs

class StackFrames(gym.ObservationWrapper):
    """
    Wrapper to stack 'm' recent frames
    """
    def __init__(self, env, repeat):
        """
        Init Function
        :param env: Input environment to wrap
        :param repeat: Number of frames to stack
        """
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis=0),
                            env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)  # Update observation_space
        self.stack = collections.deque(maxlen=repeat)  # Deque to hold 'm' observations

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)  # Update stack with the latest observation
        return np.array(self.stack).reshape(self.observation_space.low.shape)

def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env
