import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

np.random.seed(5)


def td_0(episode, step_size, benchmark):
    state_values = np.full(7, fill_value=0.5)
    state_values[6] = 1
    state_values[0] = 0
    error_list = []
    for i in range(episode):
        state = 3
        while state not in [0,6]:
            action = np.random.randint(2)
            if action == 0:
                action -= 1
            new_state = state + action
            state_values[state] = state_values[state] + step_size * (state_values[new_state] - state_values[state])
            state = new_state
        error = rms(state_values, benchmark)
        error_list.append(error)
    return state_values, np.array(error_list)


def batch_td(episode,step_size, benchmark):
    state_values = np.full(7, fill_value=-1.0)
    state_values[6] = 1
    state_values[0] = 0
    error_list = []
    trajectories = [] # 记录所有episode的trajectory
    for epi in trange(episode):
        state = 3
        trajectory = [3]
        while True:
            action = np.random.randint(2)
            if action == 0:
                action -= 1
            state += action
            trajectory.append(state)
            if state in [0,6]:
                break
        trajectories.append(trajectory)
        while True:
            increment = np.zeros(7)
            # 把之前经历的所有trajectories当成一个batch，重复进行update 直到变化量缩小到一定水平
            for trajectory in trajectories:
                for i, state in enumerate(trajectory):
                    if i < len(trajectory)-1:
                        increment[state] += state_values[trajectory[i+1]] - state_values[state]
            increment *= step_size
            if np.sum(np.abs(increment))< 1e-3:
                break
            state_values += increment
        error = rms(state_values, benchmark)
        error_list.append(error)
    return state_values, np.array(error_list)


def monte_carlo(episode, step_size, benchmark):
    state_values = np.full(7, fill_value=0.5)
    state_values[6] = 1
    state_values[0] = 0
    error_list = []

    for i in range(episode):
        state = 3
        reward = 0
        trajectory = []
        while state not in [0,6]:
            trajectory.append(state)
            action = np.random.randint(2)
            if action == 0:
                action -= 1
            new_state = state + action
            if new_state==6:
                reward += 1
            state = new_state
        for state in trajectory:
            state_values[state] += step_size * (reward - state_values[state])
        error = rms(state_values, benchmark)
        error_list.append(error)
    return state_values, error_list


def rms(state, benchmark):
    return np.average(np.sqrt((state - benchmark) ** 2))


def example_62_left():
    episodes = [0,1,10,100]
    # episode_results = []
    for episode in episodes:
        temp = td_0(episode)
        plt.plot(temp[1:-1], label=str(episode) + ' episodes')
        # episode_results.append(temp)
    plt.show()


def example_62_right(benchmark):
    episode = 100
    runs = 100
    td_alpha = [0.05,0.1,0.15]
    mc_alpha = [0.01,0.02,0.03,0.04]
    t_error = np.zeros((100,100,3))
    mc_error = np.zeros((100,100,4))
    for i in range(runs):
        for idx,ta in enumerate(td_alpha):
            x, episode_errors = td_0(episode,ta, benchmark)
            t_error[i,:,idx] = episode_errors
        for idx, ma in enumerate(mc_alpha):
            _, mc_episode_errors = monte_carlo(episode, ma, benchmark)
            mc_error[i,:,idx] = mc_episode_errors
    t_error_mean = np.average(t_error, axis=0)
    mc_error_mean = np.average(mc_error, axis=0)
    plt.plot(t_error_mean)
    plt.show()
    plt.plot(mc_error_mean)
    plt.show()
    # return t_error_mean, mc_error_mean


if __name__ == '__main__':
    # example_62_left()
    benchmark = np.arange(7)/6
    # example_62_right(benchmark)
    x,y = batch_td(100,0.001, benchmark)