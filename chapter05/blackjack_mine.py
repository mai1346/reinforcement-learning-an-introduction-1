# dealer: <17 hit, >=17 stick
# player: <20 hit, >=20 stick

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange


np.random.seed(5)


def get_a_card():
    card = np.random.randint(1,14)
    if card == 1:
        value = 11
    else:
        value = min(10,card)
    return value


def initialize():
    player_sum = 0
    dealer_sum = 0
    usable = 0
    dealer_usable = 0
    while player_sum < 12:
        card = get_a_card()
        usable += card==11
        player_sum += card
        if player_sum > 21:
            player_sum -= 10
            usable -= 1
    dealer_show = get_a_card()
    dealer_sum += dealer_show
    dealer_usable += dealer_show==11
    dealer_hide = get_a_card()
    dealer_sum += dealer_hide
    dealer_usable += dealer_hide==11
    if dealer_sum > 21:
        dealer_sum -= 10
        dealer_usable -= 1
    return player_sum, dealer_sum, dealer_show, usable, dealer_usable


def single_episode(player_s, dealer_sum, dealer_show, usable, dealer_usable):
    intermediate_player_s = []
    intermediate_dealer_s = []
    intermediate_player_usable = []

    # player move
    while player_s < 20:
        intermediate_player_s.append(player_s)
        intermediate_dealer_s.append(dealer_show)
        intermediate_player_usable.append(usable)  # 每个中间状态都要储存
        card = get_a_card()
        usable += card==11
        player_s += card
        if player_s > 21:
            if usable:
                player_s -= 10
                usable -= 1
                continue
            reward = -1
            return reward, intermediate_player_s, intermediate_dealer_s, intermediate_player_usable
    intermediate_player_s.append(player_s)
    intermediate_dealer_s.append(dealer_show)
    intermediate_player_usable.append(usable)  # 对于直接是20，21的state 要直接储存当前state
    # dealer move
    while dealer_sum < 17:
        d_card = get_a_card()
        dealer_usable += d_card==11
        dealer_sum += d_card
        if dealer_sum > 21:
            if dealer_usable:
                dealer_sum -= 10
                dealer_usable -= 1
                continue
            reward = 1
            return reward, intermediate_player_s, intermediate_dealer_s, intermediate_player_usable
    if player_s > dealer_sum:
        reward = 1
    elif player_s < dealer_sum:
        reward = -1
    else:
        reward = 0
    return reward, intermediate_player_s, intermediate_dealer_s, intermediate_player_usable


def main(state_values, state_counts):
    for i in trange(10000):
        player_sum, dealer_sum, dealer_show, usable, dealer_usable = initialize()
        reward, inter_player_s, inter_dealer_s, usable = single_episode(player_sum, dealer_sum, dealer_show, usable, dealer_usable)
        print(usable)
        player_idx_arr = np.array(inter_player_s) - 12
        dealer_idx_arr = np.array(inter_dealer_s)
        dealer_idx_arr[dealer_idx_arr==11] = 1  # 让11 重置为1
        dealer_idx_arr -= 1
        state_counts[usable, player_idx_arr, dealer_idx_arr] += 1
        state_values[usable, player_idx_arr, dealer_idx_arr] += reward
    result = state_values / state_counts
    return result[0], result[1]


if __name__ == '__main__':
    player_state = np.arange(12, 21 + 1)
    dealer_state = np.arange(1, 10 + 1)
    state_values = np.zeros((2, len(player_state), len(dealer_state)))  # 初始化一个三维矩阵，分别储存usable 和 no usable
    state_counts = np.ones((2, len(player_state), len(dealer_state)))
    main(state_values, state_counts)
    # sns.heatmap(state_values)




