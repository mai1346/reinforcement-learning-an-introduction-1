import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

np.random.seed(5)


def epsilon_greedy(state, q, epsilon):
    if np.random.random() < epsilon:
        action_idx = np.random.randint(2)
    else:
        _max = np.max(q[state, :])
        action_idx = np.random.choice(np.where(q[state, :]==_max)[0])
    return action_idx


def get_reward(state):
    if state == 2:
        reward = 0
    elif state == 0:
        reward = np.random.normal(-0.1)
    return reward


def q_learning(q_value, epsilon, stepsize):
    state = 1
    left_count = 0
    while True:
        if state == 0:
            new_state = 2
            action = np.random.randint(2)
            q_value[state, action] += stepsize * (np.random.normal(-0.1) + max(q_value[new_state, :]) - q_value[state, action])
        else:
            action = epsilon_greedy(state, q_value, epsilon)
            if action == 0:
                left_count += 1
            new_state = state + action if action==1 else state - 1
            # reward = get_reward(new_state)
            q_value[state, action] += stepsize * (0 + max(q_value[new_state, :]) - q_value[state, action])
        if new_state == 2:
            break
        state = new_state
    return left_count


def double_q(q1, q2, epsilon, stepsize):
    state = 1
    left_count = 0
    while True:
        if state == 0:
            new_state = 2
            action = np.random.randint(2)
            if np.random.random() < 0.5:
                q1[state, action] += stepsize * (np.random.normal(-0.1) + q2[new_state, np.argmax(q1[new_state, :])] - q1[state, action])
            else:
                q2[state, action] += stepsize * (np.random.normal(-0.1) + q1[new_state, np.argmax(q2[new_state, :])] - q2[state, action])
        else:
            action = epsilon_greedy(state, q1 + q2, epsilon)
            if action == 0:
                left_count += 1
            new_state = state + action if action==1 else state - 1
            # reward = get_reward(new_state)
            if np.random.random() < 0.5:
                q1[state, action] += stepsize * (0 + q2[new_state, np.argmax(q1[new_state, :])] - q1[state, action])
            else:
                q2[state, action] += stepsize * (0 + q1[new_state, np.argmax(q2[new_state, :])] - q2[state, action])
        if new_state ==2:
            break
        state = new_state
    return left_count


def figure_65(epsilon, stepsize):
    runs = 1000
    episodes = 300
    q_actions = np.zeros((runs, episodes))
    dq_actions = np.zeros((runs, episodes))
    for run in trange(runs):
        q_value = np.zeros((3, 2))
        q1 = np.zeros((3, 2))
        q2 = np.zeros((3, 2))
        for episode in range(episodes):
            q_action = q_learning(q_value, epsilon, stepsize)
            dq_action = double_q(q1, q2, epsilon, stepsize)
            q_actions[run, episode] = q_action
            dq_actions[run, episode] = dq_action
    q_left_ratio = q_actions.mean(axis=0)
    dq_left_ratio = dq_actions.mean(axis=0)
    plt.plot(q_left_ratio, label='q_learning')
    plt.plot(dq_left_ratio,label='dq_learning')
    plt.legend()
    plt.show()
    return q_actions, dq_actions
    # return q_actions


if __name__ == '__main__':
    epsilon = 0.1
    stepsize = 0.1
    # q_value = np.zeros((3, 2))
    # q1 = np.zeros((3, 2))
    # q2 = np.zeros((3, 2))
    test = figure_65(epsilon, stepsize)
