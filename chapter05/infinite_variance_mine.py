import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from infinite_variance import play
stay_p = 0.9

np.random.seed(1346)

def pure_random():
    temp = np.random.randint(0, 2)
    return temp

def single_episode(policy):
    action = policy()
    action_trajectory = []
    while action == 0:
        action_trajectory.append(action)
        if np.random.random() < stay_p:
            action = policy()
        else:
            return 1, action_trajectory
    action_trajectory.append(action)
    return 0, action_trajectory


def play():
    # track the action for importance ratio
    trajectory = []
    while True:
        action = pure_random()
        trajectory.append(action)
        if action == 1:
            return 0, trajectory
        if np.random.binomial(1, 0.9) == 0:
            return 1, trajectory


def off_policy(episode):
    values = np.zeros(episode)
    # sample_importance_sum = np.zeros(episode)
    runs = 10
    for run in trange(runs):
        for i in range(episode):
            reward,actions = single_episode(pure_random)
            if actions[-1] == 1:
                ratio = 0
            else:
                ratio = 1 / 0.5**(len(actions))
            reward *= ratio
            values[i] = reward
            # sample_importance_sum[i] = ratio
        values = values.cumsum()
        ordinary_sampling = values / np.arange(1, episode + 1)
        plt.plot(ordinary_sampling)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.show()


    # return ordinary_sampling

if __name__ == '__main__':
    off_policy(100000)