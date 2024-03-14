
# Q table learning:
# Q(s,a) ← Q(s,a)+α(r+γmax a′Q(s′,a′))

# Q함수는 reward + 다음상태에 대한 Q함수의 최대값이 되게 갱신하고,
# 미래의 Q함수의 최대값은 epsilon-greedy방법으로 이미 경험한 reward일 가능성을 고려하여 현재 Q의 최대값이나 랜덤값중에 확률적으로 선택한다. 

# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap


import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

"""
FrozenLake solver using Q-table
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap
"""

import time

import gym
import numpy as np

import utils.prints as print_utils

N_ACTIONS = 4
N_STATES = 16

LEARNING_RATE = .5
DISCOUNT_RATE = .98

N_EPISODES = 2000

def main():
    """Main"""
    # env = gym.make("FrozenLake-v0")
    env = gym.make("LunarLander-v2", render_mode="human")

    # Initialize table with all zeros
    Q = np.zeros([N_STATES, N_ACTIONS])

    # Set learning parameters

    # create lists to contain total rewards and steps per episode
    rewards = []

    for i in range(N_EPISODES):
        # Reset environment and get first new observation
        state = env.reset()
        episode_reward = 0
        done = False

        # The Q-Table learning algorithm
        while not done:
            # Choose an action by greedily (with noise) picking from Q table
            noise = np.random.randn(1, N_ACTIONS) / (i + 1)
            action = np.argmax(Q[state, :] + noise)

            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)

            reward = -1 if done and reward < 1 else reward

            # Update Q-Table with new knowledge using learning rate
            Q[state, action] = (
                1 - LEARNING_RATE) * Q[state, action] + LEARNING_RATE * (
                    reward + DISCOUNT_RATE * np.max(Q[new_state, :]))

            episode_reward += reward
            state = new_state

        rewards.append(episode_reward)

    print("Score over time: " + str(sum(rewards) / N_EPISODES))
    print("Final Q-Table Values")

    for i in range(10):
        # Reset environment and get first new observation
        state = env.reset()
        episode_reward = 0
        done = False

        # The Q-Table learning algorithm
        while not done:
            # Choose an action by greedily (with noise) picking from Q table
            action = np.argmax(Q[state, :])

            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)
            print_utils.clear_screen()
            env.render()
            time.sleep(.1)

            episode_reward += reward
            state = new_state

            if done:
                print("Episode Reward: {}".format(episode_reward))
                print_utils.print_result(episode_reward)

        rewards.append(episode_reward)

    env.close()


if __name__ == '__main__':
    main()