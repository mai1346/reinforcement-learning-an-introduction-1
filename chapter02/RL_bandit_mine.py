#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: mai1346
@version: 1.0.0
@file: RL_bandit_mine.py
@time: 2022/1/5 10:31
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time



def single_run(num_arms, time_step, epsilon):
    rewards = np.zeros(time_step)
    best_action_arr = np.zeros(time_step)
    action_mean = np.random.randn(num_arms)
    best_action = np.argmax(action_mean)
    action_value_list = np.zeros(num_arms)
    action_value_count = np.ones(num_arms)
    est_action_value = np.zeros(num_arms)
    for step in range(time_step):
        if step == 0:
            greedy_action = np.random.choice(num_arms)
        else:
            if np.random.rand() < epsilon:
                greedy_action = np.random.choice(num_arms)
            else:
                greedy_action = np.argmax(est_action_value)
        if greedy_action == best_action:
            best_action_arr[step] = 1
        # est_max = np.max(est_action_value)
        # greedy_action = np.random.choice(np.where(est_action_value==est_max)[0])
        greedy_action_value = np.random.randn() + action_mean[greedy_action]
        action_value_list[greedy_action] += greedy_action_value
        action_value_count[greedy_action] += 1
        rewards[step] = greedy_action_value
        est_action_value = action_value_list / action_value_count
    return rewards, best_action_arr


def single_run_incremental(num_arms, time_step, epsilon, fix_step=False):
    rewards = np.zeros(time_step)
    best_action_arr = np.zeros(time_step)
    action_mean = np.random.randn(num_arms)
    best_action = np.argmax(action_mean)
    # action_value_list = np.zeros(num_arms)
    action_value_count = np.ones(num_arms)
    est_action_value = np.zeros(num_arms)
    for step in range(time_step):
        if step == 0:
            greedy_action = np.random.choice(num_arms)
        else:
            if np.random.rand() < epsilon:
                greedy_action = np.random.choice(num_arms)
            else:
                greedy_action = np.argmax(est_action_value)
        if greedy_action == best_action:
            best_action_arr[step] = 1
        # est_max = np.max(est_action_value)
        # greedy_action = np.random.choice(np.where(est_action_value==est_max)[0])
        action_value_count[greedy_action] += 1
        greedy_action_value = np.random.randn() + action_mean[greedy_action]
        if fix_step:
            est_action_value[greedy_action] += (greedy_action_value - est_action_value[greedy_action]) * 0.1
        else:
            est_action_value[greedy_action] += (greedy_action_value - est_action_value[greedy_action]) / action_value_count[greedy_action]
        rewards[step] = greedy_action_value
    return rewards, best_action_arr


def single_run_non_station(num_arms, time_step, epsilon, fix_step=True, ):
    rewards = np.zeros(time_step)
    best_action_arr = np.zeros(time_step)
    action_mean = np.zeros(num_arms)

    # action_value_list = np.zeros(num_arms)
    action_value_count = np.ones(num_arms)
    est_action_value = np.zeros(num_arms)
    for step in range(time_step):
        action_mean += np.random.normal(loc=0,scale=0.01, size=num_arms)
        best_action = np.argmax(action_mean)
        if np.random.rand() < epsilon:
            greedy_action = np.random.choice(num_arms)
        else:
            greedy_action = np.argmax(est_action_value)
        if greedy_action == best_action:
            best_action_arr[step] = 1
        # est_max = np.max(est_action_value)
        # greedy_action = np.random.choice(np.where(est_action_value==est_max)[0])
        action_value_count[greedy_action] += 1
        greedy_action_value = np.random.randn() + action_mean[greedy_action]
        if fix_step:
            est_action_value[greedy_action] += (greedy_action_value - est_action_value[greedy_action]) * 0.1
        else:
            est_action_value[greedy_action] += (greedy_action_value - est_action_value[greedy_action]) / action_value_count[greedy_action]
        rewards[step] = greedy_action_value
    return rewards, best_action_arr


def single_run_non_station2(num_arms, time_step, epsilon, learn_step=None, method='e_greedy'):
    rewards = np.zeros(time_step)
    best_action_arr = np.zeros(time_step)
    action_mean = np.zeros(num_arms)
    # action_mean = np.random.randn(num_arms)

    action_value_count = np.zeros(num_arms)
    est_action_value = np.zeros(num_arms)
    # h_prefer = np.zeros(num_arms)
    baseline = 0
    for step in range(time_step):
        # non stationary change of action_mean
        action_mean += np.random.normal(loc=0,scale=0.01, size=num_arms)
        best_action = np.argmax(action_mean)
        # actual value draw from distributions located at action_means
        actual_action_values = np.random.randn(num_arms) + action_mean
        if method == 'e_greedy':
            if step == 0:
                selected_action = np.random.choice(num_arms)
            else:
                if np.random.rand() < epsilon:
                    selected_action = np.random.choice(num_arms)
                else:
                    selected_action = np.argmax(est_action_value)

        elif method == 'UCB':
            action_score = est_action_value + 2 * np.sqrt(np.log(step) / (action_value_count + 1e-5))
            selected_action = np.argmax(action_score)
        elif method == 'gradient':
            action_prob = np.exp(est_action_value) / np.exp(est_action_value).sum()
            selected_action = np.random.choice(num_arms, p=action_prob)
            one_zero = np.ones_like(action_prob)
            one_zero[selected_action] = 1
            reward = actual_action_values[selected_action]
            baseline += (reward - baseline) / step
            est_action_value += learn_step * (reward - baseline) * (one_zero - action_prob)
            # h_prefer[selected_action] += learn_step * (reward - baseline) * (1 - action_prob[selected_action])
            # h_prefer[not_selected] -= learn_step * (reward - baseline) * (action_prob[not_selected])
        else:
            raise NotImplementedError
        reward = actual_action_values[selected_action]
        action_value_count[selected_action] += 1
        if learn_step:
            est_action_value[selected_action] += (reward - est_action_value[selected_action]) * learn_step
        else:
            est_action_value[selected_action] += (reward - est_action_value[selected_action]) / action_value_count[selected_action]

        if selected_action == best_action:
            best_action_arr[step] = 1
        rewards[step] = reward

    return rewards, best_action_arr


def run_loop(runs, epsilon_list, time_step, func, **kwargs):
    reward_arr = np.zeros((len(epsilon_list), runs, time_step))
    best_action_arr = np.zeros_like(reward_arr)
    for i, epsilon in enumerate(epsilon_list):
        print("epsilon:", epsilon)
        for run in tqdm(range(runs)):
            run_reward, run_best_action = func(num_arms, time_step, epsilon, **kwargs)
            reward_arr[i, run] = run_reward
            best_action_arr[i, run] = run_best_action

    mean_best_action_counts = best_action_arr.mean(axis=1)
    mean_rewards = reward_arr.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def figure_plot(epsilon_list, mean_rewards, mean_best_action_counts):
    plt.figure(figsize=(16, 18))
    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilon_list, mean_rewards):
        plt.plot(rewards, label=f'$\epsilon = {eps}$')
        # break
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilon_list, mean_best_action_counts):
        plt.plot(counts, label=f'$\epsilon = {eps}$')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.savefig(f"{time.time()}.png")
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    num_arms = 10
    action_mean = np.random.randn(num_arms)
    time_step = 1000
    runs = 2000
    # epsilon_list = [0,0.01,0.1]
    # epsilon_list = [False, True]
    epsilon_list = [0]
    mean_best_action_counts, mean_rewards = run_loop(runs, epsilon_list, time_step, single_run_non_station2, method='UCB')
    figure_plot(epsilon_list,mean_rewards,mean_best_action_counts)