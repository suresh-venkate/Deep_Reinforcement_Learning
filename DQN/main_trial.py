import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from utils import *

# import gym

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env = gym.make('PongNoFrameskip-v4', render_mode = 'rgb_array')
    #env = gym.make('PongNoFrameskip-v4', render_mode = 'human')
    env = RepeatActionAndMaxFrame(env, 4, False, 0, False)
    env = PreprocessFrame((84,84,1), env)
    env = StackFrames(env, 4)
    env = RecordVideo(env, './Videos', episode_trigger=lambda x: x >= 1)
    obs = env.reset(seed=0, options = {})
    for i in range(1000):
        action = env.action_space.sample()
        observation_, reward, done, info, _ = env.step(action)
        print(i, action, done)
        if done:
            env.reset()

    env.close()
    print("Debug")