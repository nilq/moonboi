import gym
from deepq import Agent

import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    brain = Agent(
        gamma=0.8,
        epsilon=1.0,
        batch_size=64,
        n_actions=4,
        input_dim=[8],
        lr=0.008
    )

    scores = []
    eps_history = []
    n_games = 3500
    score = 0

    for i in range(n_games):
        if i % 10 == 0:
            avg_score = np.mean(scores[max(0, i - 10):(i + 1)])

            print(f'epoch {i} $ score {score} with average score {avg_score}')

        else:
            print(f'epoch {i} $ score {score}')
        
        score = 0
        eps_history.append(brain.epsilon)
        obs = env.reset()

        done = False

        
        while not done:
            action = brain.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            
            score += reward

            brain.store_transition(obs, action, reward, obs_, done)
            brain.learn()

            if i % 200 == 0:
                env.render()

            obs = obs_
        
        scores.append(score)
