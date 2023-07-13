import gymnasium as gym
import numpy as np
# from gymnasium.wrappers import RecordVideo

# import gym

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env = gym.make('Pong-v4', render_mode = 'rgb_array')
    #env = RecordVideo(env, './Videos', episode_trigger=lambda x: x >= 1)
    obs = env.reset()
    for i in range(10):
        # if (i % 2 == 0):
        #     action = 0
        # else:
        #     action = 1
        action = np.random.randint(0, 1)
        observation_, reward, done, info, _ = env.step(action)
        print(i, action, done)
        if done:
            env.reset()

    env.close()
    print("Debug")