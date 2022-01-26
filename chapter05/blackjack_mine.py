# dealer: <17 hit, >=17 stick
# player: <20 hit, >=20 stick

import numpy as np
from tqdm import trange

np.random.seed(5)


def get_a_card():
    card = np.random.choice(13)+1
    value = min(10,card)
    return value


def initialize():
    player_sum = 0
    dealer_sum = 0
    while player_sum < 12:
        player_sum += get_a_card()
    dealer_show = get_a_card()
    dealer_sum += dealer_show
    dealer_sum += get_a_card()
    return player_sum, dealer_sum, dealer_show


def single_episode(player_s, dealer_sum, dealer_show):
    intermediate_player_s = []
    intermediate_dealer_s = []
    # player move
    while player_s < 20:
        intermediate_player_s.append(player_s)
        intermediate_dealer_s.append(dealer_show)
        player_s += get_a_card()
        if player_s > 21:
            reward = -1
            print(f"player:{player_s};dealer:{dealer_show}; reward:{reward}")
            return reward, intermediate_player_s, intermediate_dealer_s
        # elif player_s in [20,21]:
        #     break
    intermediate_player_s.append(player_s)
    intermediate_dealer_s.append(dealer_show)
    # dealer move
    while dealer_sum < 17:
        dealer_sum += get_a_card()
        if dealer_sum > 21:
            reward = 1
            print(f"player:{player_s};dealer:{dealer_show}; reward:{reward}")
            return reward, intermediate_player_s, intermediate_dealer_s
        # elif dealer_sum >= 17:
        #     break
    if player_s > dealer_sum:
        reward = 1
    elif player_s < dealer_sum:
        reward = -1
    else:
        reward = 0
    print(f"player:{player_s};dealer:{dealer_show}; reward:{reward}")
    return reward, intermediate_player_s, intermediate_dealer_s


if __name__ == '__main__':
    # state = [10, 1, 9]
    player_state = np.arange(12, 21 + 1)
    dealer_state = np.arange(1, 10 + 1)
    # player_usable = np.arange(0,1+1)
    state_values = np.zeros((len(player_state), len(dealer_state)))
    state_counts = np.zeros((len(player_state), len(dealer_state)))

    for i in trange(10000):
        player_sum, dealer_sum, dealer_show = initialize()
        reward, inter_player_s, inter_dealer_s = single_episode(player_sum, dealer_sum, dealer_show)
        # print(reward)
        player_idx_arr = np.array(inter_player_s) - 12
        dealer_idx_arr = np.array(inter_dealer_s) - 1
        state_counts[player_idx_arr, dealer_idx_arr] += 1
        state_values[player_idx_arr, dealer_idx_arr] += reward

    state_values /= state_counts




