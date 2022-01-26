# dealer: <17 hit, >=17 stick
# player: <20 hit, >=20 stick

import numpy as np
from tqdm import trange


def get_a_card():
    card = np.random.choice(13)+1
    value = min(10,card)
    return value


def initialize():
    player_sum = 0
    dealer_sum = 0
    player_sum += get_a_card()
    player_sum += get_a_card()
    dealer_show = get_a_card()
    dealer_sum += dealer_show
    dealer_sum += get_a_card()
    return player_sum, dealer_sum, dealer_show


def single_episode(player_s, dealer_show, dealer_sum):
    intermediate_player_s = [player_s]
    intermediate_dealer_s = [dealer_show]
    # player move
    while player_s < 20:
        player_s += get_a_card()
        intermediate_player_s.append(player_s)
        intermediate_dealer_s.append(dealer_show)
        if player_s > 21:
            reward = -1
            return reward, intermediate_player_s, intermediate_dealer_s
        elif player_s in [20,21]:
            break
    # dealer move
    while dealer_sum < 17:
        dealer_sum += get_a_card()
        if dealer_sum > 21:
            reward = 1
            return reward, intermediate_player_s, intermediate_dealer_s
        elif dealer_sum >= 17:
            break

    if player_s > dealer_sum:
        reward = 1
        return reward, intermediate_player_s, intermediate_dealer_s
    elif player_s < dealer_sum:
        reward = -1
        return reward, intermediate_player_s, intermediate_dealer_s
    else:
        reward = 0
        return reward, intermediate_player_s, intermediate_dealer_s


if __name__ == '__main__':
    # state = [10, 1, 9]
    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    # player_usable = np.arange(0,1+1)
    state_values = np.zeros((len(dealer_show), len(player_sum)))
    for p_sum in player_sum:
        for d_show in dealer_show:
            state = [p_sum, d_show]
            print(state)
            state_sum = 0
            for i in trange(10000):
                state_sum += single_episode(state)
            state_value = state_sum / 10000
            state_values[d_show-10,p_sum-12] = state_value

    # single_episode([20, 1])



