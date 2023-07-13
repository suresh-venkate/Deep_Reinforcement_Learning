import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gym import wrappers

if __name__ == '__main__':
    # Note: Environment name should be 'PongNoFrameskip-v4' and not the regular one
    # to avoid the following issues
    #   - Avoid random action repeat and random actions    #   -
    env = make_env('PongNoFrameskip-v4')  # Instantiate Pong env.
    best_score = -np.inf  # Placeholder for best_score. Initialize to -inf.
    load_checkpoint = False  # Flag to indicate whether to load checkpoint or not
    n_games = 1  # Number of games to play during training

    # Instantiate DQN agent
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=500, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    # fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
    #         + str(n_games) + 'games'
    # figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    # env = wrappers.Monitor(env, "tmp/dqn-video", video_callable=lambda episode_id: True, force=True)
    # env = RecordVideo(env, "./tmp/dqn-video", episode_trigger=lambda x: x >= 1)
    n_steps = 0  # Number of environment steps taken so far
    # Define a few placeholders
    # scores array: Stores the score of each episode. Score is the cumulative undiscounted
    # reward over the episode
    # eps_history: Array to store epsilon value at the end of each episode.
    # steps_array: Array to store the number of steps taken during each episode.
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):  # Iterate through n_games
        done = False  # Initialize 'done' flag to false
        score = 0  # Initialize score to 0.
        observation = env.reset()  # Reset environment

        while not done:  # Loop through episode till termination
            action = agent.choose_action(observation)  # Obtain action through eps-greedy action seln.
            observation_, reward, done, info = env.step(action)  # Take env step
            score += reward  # Update score with latest reward - Score is the cumulative undiscounted
                             # reward over the episode
            if not load_checkpoint:  # Check whether we are in training mode
                # Store current sample in replay memory
                agent.store_transition(observation, action, reward, observation_, done)
                #agent.learn()  # Perform learning step
            observation = observation_  # S(t+1) = St
            n_steps += 1  # Update n_steps after every env. step
            print("Running step %d; Current Score: %d" %(n_steps, score))
        scores.append(score)  # Update 'scores' array
        steps_array.append(n_steps)  # Update 'steps_array'

    #     avg_score = np.mean(scores[-100:])
    #     print('episode: ', i,'score: ', score,
    #          ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
    #         'epsilon %.2f' % agent.epsilon, 'steps', n_steps)
    #
    #     if avg_score > best_score:
    #         if not load_checkpoint:
    #             agent.save_models()
    #         best_score = avg_score
    #
    #     eps_history.append(agent.epsilon)
    #
    # x = [i+1 for i in range(len(scores))]
    # plot_learning_curve(steps_array, scores, eps_history, figure_file)
