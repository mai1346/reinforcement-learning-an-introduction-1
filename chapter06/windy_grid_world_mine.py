import numpy as np
from tqdm import trange

from chapter04.car_rental_mine import cartesian_prod

np.random.seed(5)


class WindyWorld(object):

    def __init__(self, hight, width, start, end, wind_force):
        self.hight = hight
        self.width = width
        self.start = start
        self.end = end
        self.init_wind_force = wind_force.copy()
        self.wind_force = wind_force.copy()

    def reset_wind_force(self):
        self.wind_force = self.init_wind_force

    def stochastic_wind(self):
        self.reset_wind_force()
        random_wind = np.random.randint(-1,2,len(self.wind_force))
        random_wind = np.where(self.wind_force==0,0,random_wind)
        self.wind_force += random_wind


def action_gen(kings_move=True):
    vertical_possible = np.arange(-1, 2)
    horizontal_possible = np.arange(-1, 2)
    actions = cartesian_prod(vertical_possible, horizontal_possible)
    actions = np.vstack(actions).T
    if kings_move is None:
        return actions
    mask = np.any(actions != 0, axis=1)
    if not kings_move:
        mask2 = np.any(abs(actions)!=1, axis=1)
        mask &= mask2
    return actions[mask]


def move(env: WindyWorld, state, action):
    new_state_vertical = state[0] + action[0] - env.wind_force[state[1]]
    new_state_horizontal = state[1] + action[1]
    new_state_vertical = np.clip(new_state_vertical, 0, env.hight-1)
    new_state_horizontal = np.clip(new_state_horizontal, 0, env.width-1)
    return new_state_vertical, new_state_horizontal


def epsilon_greedy(state, actions, q, epsilon):
    if np.random.random() < epsilon:
        action_idx = np.random.randint(len(actions))
    else:
        action_idx = np.argmax(q[state[0], state[1],:])
    return action_idx


def single_episode(env: WindyWorld, actions, epsilon, step_size, q, stochastic_wind=False):
    state = env.start
    ending = False
    action_idx = epsilon_greedy(state, actions, q, epsilon)
    steps = 0
    trajectory = [state]
    while not ending:
        if stochastic_wind:
            env.stochastic_wind()
        new_state = move(env, state, actions[action_idx])
        new_action_idx = epsilon_greedy(new_state, actions, q, epsilon)
        q[state+(action_idx,)] += step_size * (-1 + q[new_state+(new_action_idx,)] - q[state+(action_idx,)])
        if new_state == env.end:
            break
        state = new_state
        action_idx = new_action_idx
        steps+=1
        trajectory.append(state)
    return steps, trajectory


if __name__ == '__main__':
    epsilon = 0.1
    step_size = 0.5
    # actions
    actions = action_gen(kings_move=True)
    # gridworld
    hight = 7
    width = 10
    start = (3, 0)
    end = (3, 7)
    wind_force = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
    world = WindyWorld(hight=hight, width=width, start=start, end=end, wind_force=wind_force)
    q = np.zeros((hight, width, len(actions)))
    episodes = 200
    for episode in trange(episodes):
        single_episode(world, actions, epsilon, step_size, q, stochastic_wind=True)

    world.reset_wind_force()
    steps = single_episode(world, actions, 0, step_size, q, stochastic_wind=True)






