# dealer: <17 hit, >=17 stick
# player: <20 hit, >=20 stick

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

# np.random.seed(5)


def get_a_card():
    card = np.random.randint(1, 14)
    if card == 1:
        value = 11
    else:
        value = min(10, card)
    return value


# 定义policy：对于21点游戏，policy即player面对不同的state时，hit和stand的概率，用0代表stick，1代表hit。不同的policy依赖的state状态不
# 一样。
def stick_above(player_sum, dealer_show, usable, state_values, state_counts):
    assert player_sum <= 21
    if player_sum < 20:
        return 1
    return 0


def pure_random(player_sum, dealer_show, usable, state_values, state_counts):
    temp = np.random.randint(0, 2)
    return temp


def pure_random2(player_sum, dealer_show, usable, state_values, state_counts):
    if np.random.binomial(1, 0.5) == 1:
        return 0
    return 1


def greedy(player_sum, dealer_show, usable, state_values, state_counts):
    player_sum -= 12
    dealer_show -= 11 if dealer_show == 11 else 1
    action_values = state_values[player_sum, dealer_show, usable, :] / state_counts[player_sum, dealer_show, usable, :]
    # action = np.argmax(action_values)
    action = np.random.choice(np.where(action_values == np.max(action_values))[0])
    return action


def initialize(method='random'):
    if method == 'random':
        dealer_usable = 0
        player_sum = np.random.randint(12, 22)
        dealer_show = np.random.randint(2, 12)
        usable = np.random.randint(0, 2)
        dealer_usable += dealer_show == 11
        dealer_sum = dealer_show + np.random.randint(2, 12)
        dealer_usable += dealer_sum == 22
    elif method == 'fixed_state':
        player_sum = 13
        usable = 1
        dealer_show = 2
        dealer_usable = 0
        dealer_sum = dealer_show + np.random.randint(2, 12)
        dealer_usable += dealer_sum == 22
    else:
        player_sum = 0
        dealer_sum = 0
        usable = 0
        dealer_usable = 0
        while player_sum < 12:
            card = get_a_card()
            usable += card == 11
            player_sum += card
            if player_sum > 21:
                player_sum -= 10
                usable -= 1
        dealer_show = get_a_card()
        dealer_sum += dealer_show
        dealer_usable += dealer_show == 11
        dealer_hide = get_a_card()
        dealer_sum += dealer_hide
        dealer_usable += dealer_hide == 11
    if dealer_sum > 21:
        dealer_sum -= 10
        dealer_usable -= 1
    return player_sum, dealer_sum, dealer_show, usable, dealer_usable


def single_episode(player_s, dealer_sum, dealer_show, usable, dealer_usable, state_values, state_counts,
                   policy=stick_above, random_initial_action=False):
    """

    @param player_s:
    @param dealer_sum:
    @param dealer_show:
    @param usable:
    @param dealer_usable:
    @param state_values:
    @param state_counts:
    @param policy:
    @param random_initial_action:
    @return:
    """
    intermediate_player_s = []
    intermediate_dealer_s = []
    intermediate_player_usable = []
    intermediate_action = []
    hit = np.random.randint(0, 2) if random_initial_action else policy(player_s, dealer_show, usable, state_values,
                                                                       state_counts)
    # player move
    while hit:
        intermediate_player_s.append(player_s)
        intermediate_dealer_s.append(dealer_show)
        intermediate_action.append(hit)
        intermediate_player_usable.append(usable)  # 每个中间状态都要储存
        card = get_a_card()
        usable += card == 11
        player_s += card
        while usable and player_s > 21:
            player_s -= 10
            usable -= 1
        if player_s > 21:
            reward = -1
            return reward, intermediate_player_s, intermediate_dealer_s, intermediate_player_usable, intermediate_action
        else:
            hit = policy(player_s, dealer_show, usable, state_values, state_counts)
    intermediate_player_s.append(player_s)
    intermediate_dealer_s.append(dealer_show)
    intermediate_action.append(hit)
    intermediate_player_usable.append(usable)  # 对于直接是20，21的state 要直接储存当前state
    # dealer move
    while dealer_sum < 17:
        d_card = get_a_card()
        dealer_usable += d_card == 11
        dealer_sum += d_card
        if dealer_sum > 21:
            if dealer_usable:
                dealer_sum -= 10
                dealer_usable -= 1
                continue
            reward = 1
            return reward, intermediate_player_s, intermediate_dealer_s, intermediate_player_usable, intermediate_action
    if player_s > dealer_sum:
        reward = 1
    elif player_s < dealer_sum:
        reward = -1
    else:
        reward = 0
    return reward, intermediate_player_s, intermediate_dealer_s, intermediate_player_usable, intermediate_action


def monte_carlo_policy_eval(num_eps):
    state_values = np.zeros((2, len(player_state), len(dealer_state)))  # 初始化一个三维矩阵，分别储存usable 和 no usable
    state_counts = np.ones((2, len(player_state), len(dealer_state)))
    for i in trange(num_eps):
        player_sum, dealer_sum, dealer_show, usable, dealer_usable = initialize()
        reward, inter_player_s, inter_dealer_s, usable, action = single_episode(player_sum, dealer_sum, dealer_show,
                                                                                usable, dealer_usable, state_values,
                                                                                state_counts)
        # print(usable)
        player_idx_arr = np.array(inter_player_s) - 12
        dealer_idx_arr = np.array(inter_dealer_s)
        dealer_idx_arr[dealer_idx_arr == 11] = 1  # 让11 重置为1
        dealer_idx_arr -= 1
        state_counts[usable, player_idx_arr, dealer_idx_arr] += 1
        state_values[usable, player_idx_arr, dealer_idx_arr] += reward
    result = state_values / state_counts
    return result[0], result[1]


def monte_carlo_es(num_eps, exploring_starts=False):
    state_values = np.zeros((len(player_state), len(dealer_state), 2, 2))  # 初始化一个4维矩阵，分别储存
    state_counts = np.ones((len(player_state), len(dealer_state), 2, 2))
    for i in trange(num_eps):
        # print(i)
        player_sum, dealer_sum, dealer_show, usable, dealer_usable = initialize(random=exploring_starts)
        policy = greedy if i else stick_above
        reward, inter_player_s, inter_dealer_s, usable, action = single_episode(player_sum, dealer_sum, dealer_show,
                                                                                usable, dealer_usable, state_values,
                                                                                state_counts, policy=policy,
                                                                                random_initial_action=True)
        player_idx_arr = np.array(inter_player_s) - 12
        dealer_idx_arr = np.array(inter_dealer_s)
        dealer_idx_arr[dealer_idx_arr == 11] = 1  # 让11 重置为1
        dealer_idx_arr -= 1
        state_counts[player_idx_arr, dealer_idx_arr, usable, action] += 1
        state_values[player_idx_arr, dealer_idx_arr, usable, action] += reward
    return state_values / state_counts


def monte_carlo_off_policy(episode_num):
    """
    实现off policy的蒙特卡洛模拟，weighted_average控制importance sampling 的方式，
    True: weighted_averaged sampling; False: ordinary sampling
    针对一个固定的state：
    dealer_show: 2
    player_sum: 13
    usable: 1
    behavior_policy: pure_random
    target_policy: stick_above(20)
    @return:
    """
    state_value = np.zeros(episode_num)
    sample_importance_sum = np.zeros(episode_num)
    for i in range(episode_num):
        player_sum, dealer_sum, dealer_show, usable, dealer_usable = initialize('fixed_state')
        reward, player_sums, dealer_shows, usable, actions = single_episode(player_sum, dealer_sum, dealer_show, usable,
                                                                            dealer_usable, state_values=None,
                                                                            state_counts=None, policy=pure_random)
        target_action_prob = 1
        episode_action_prob = 1
        for action, player_s in zip(actions, player_sums):
            if stick_above(player_s, dealer_show, usable, state_value, state_counts=None) == action:
                episode_action_prob *= 0.5
            else:
                target_action_prob = 0
                break
        sample_importance_ratio = target_action_prob / episode_action_prob
        reward *= sample_importance_ratio
        state_value[i] = reward
        sample_importance_sum[i] = sample_importance_ratio
    state_value = state_value.cumsum()
    ordinary_sampling = state_value / np.arange(1, episode_num + 1)
    rhos = sample_importance_sum.cumsum()
    with np.errstate(divide='ignore',invalid='ignore'):
        weighted_average_sampling = np.where(rhos != 0, state_value / rhos, 0)
    # weighted_average_sampling = state_value / sample_importance_sum.cumsum()
    return ordinary_sampling, weighted_average_sampling


def plot_5_2():
    test = monte_carlo_es(500000, exploring_starts=True)
    state_value_no_usable_ace = np.max(test[:, :, 0, :], axis=2)
    state_value_usable_ace = np.max(test[:, :, 1, :], axis=2)

    action_no_usable_ace = 1 - np.argmax(test[:, :, 0, :], axis=2)
    action_usable_ace = 1 - np.argmax(test[:, :, 1, :], axis=2)
    # sns.heatmap(state_values)
    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)
    plt.show()
    # plt.savefig('../images/figure_5_2.png')
    plt.close()


def figure_5_3():
    true_value = -0.27726
    runs = 100
    episodes = 10000
    error_ordinary = np.zeros(episodes)
    error_weighted = np.zeros(episodes)
    for i in trange(runs):
        ordinary_sampling_, weighted_sampling_ = monte_carlo_off_policy(episodes)
        # get the squared error
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(np.arange(1, episodes + 1), error_ordinary, color='green', label='Ordinary Importance Sampling')
    plt.plot(np.arange(1, episodes + 1), error_weighted, color='red', label='Weighted Importance Sampling')
    plt.ylim(-0.1, 5)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel(f'Mean square error\n(average over {runs} runs)')
    plt.xscale('log')
    plt.legend()
    plt.show()
    # plt.savefig('../images/figure_5_3.png')
    plt.close()
    return error_ordinary, error_weighted


if __name__ == '__main__':
    player_state = np.arange(12, 21 + 1)
    dealer_state = np.arange(1, 10 + 1)
    o, w = figure_5_3()