import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def space_1():
    space = np.ones((32,17))
    space[7:,9:] = 0
    space[:3,:2] = 0
    space[3,0] = 0
    space[14:,0] = 0
    space[22:,1] = 0
    space[0,2] = 0
    space[-3:,2] = 0
    space[6,10:] = 0

    start_x = np.array([31])
    start_y = np.arange(3,9)
    finish_x = np.arange(6)
    finish_y = np.array([16])
    return space, start_x, start_y, finish_x,finish_y


def random_policy():
    vertical = np.random.randint(-1,2)
    horizontal = np.random.randint(-1,2)
    return vertical, horizontal


def greedy_policy(space, start_xs, start_ys, end_xs, end_ys, state_values, state):
    actions_values = np.zeros((3,3))
    for col,h in enumerate(range(-1,2)):
        for row, v in enumerate(range(-1,2)):
            new_x_vel = state[2] - v
            new_y_vel = state[3] + h
            new_x = state[0] - new_x_vel
            new_y = state[1] - new_y_vel
            if in_road(new_x, new_y, space):
                new_state_value = state_values[new_x, new_y, new_x_vel, new_y_vel]
            elif is_finish(new_x, new_y, end_xs, end_ys):
                new_state_value = state_values[new_x, 16, new_x_vel, new_y_vel]
            else:
                sx, sy, _ = start_init(start_xs, start_ys)
                new_state_value = state_values[sx, sy, 0, 0]
            # new_state_value = state_values[new_x, new_y, new_x_vel, new_y_vel]
            actions_values[row, col] = new_state_value
    print(actions_values)
    index = np.unravel_index(np.argmax(actions_values),(3,3))
    return index[0]-1, index[1]-1


def is_finish(x, y, end_xs, end_ys):
    """
    是否越过终点线，并不是停留在终点线上，而是应该x属于end_xs,而y>= end_ys
    @param x:
    @param y:
    @param end_xs:
    @param end_ys:
    @return:
    """
    if x in end_xs and y >= end_ys[0]:
        return True
    else:
        return False


def in_road(x, y, space):
    if x < 0 or y < 0:
        return False
    elif x >= space.shape[0] or y >= space.shape[1]:
        return False
    elif space[x, y]:
        return True
    else:
        return False


def start_init(start_xs, start_ys):
    xloc = np.random.randint(start_xs[0], start_xs[-1]+1)
    yloc = np.random.randint(start_ys[0], start_ys[-1]+1)
    velocity = [0,0]
    return xloc, yloc, velocity


def make_trajectory(space, start_xs, start_ys, end_xs, end_ys, epsilon=0.1):
    xloc, yloc, velocity = start_init(start_xs, start_ys)
    trajectory = []
    while not is_finish(xloc, yloc, end_xs, end_ys):
        # print(velocity)
        if np.random.random() <= epsilon:
            x_delta, y_delta = 0,0
        else:
            x_delta, y_delta = random_policy()
        velocity[0] = max(min(4, velocity[0]+x_delta), 0)
        velocity[1] = max(min(4, velocity[1]+y_delta), 0)
        xloc -= velocity[0]
        yloc += velocity[1]
        if not in_road(xloc, yloc, space) and not is_finish(xloc, yloc, end_xs, end_ys):
            xloc, yloc, velocity = start_init(start_xs, start_ys)
        trajectory.append((xloc, yloc, velocity[0], velocity[1], x_delta, y_delta))
    return np.array(trajectory)


def off_policy(episodes):
    """
    every_visit
    @param episodes:
    @return:
    """
    space, start_x, start_y, finish_x, finish_y = space_1()
    state_value = np.full((space.shape[0], space.shape[1], 5, 5), fill_value=0.0)
    cum_ratio = np.zeros_like(state_value, dtype='float')
    for i in trange(episodes):
        trajectory = make_trajectory(space, start_x, start_y, finish_x, finish_y)
        reward = 0
        ratio = 1
        for (xloc, yloc, x_v, y_v, x, y) in trajectory:
            # print(reward)
            reward += -1
            cum_ratio[xloc, yloc, x_v, y_v] += ratio
            state_value[xloc, yloc, x_v, y_v] += ratio * (reward - state_value[xloc, yloc, x_v, y_v]) / cum_ratio[xloc, yloc,x_v, y_v]
            if (x, y) != greedy_policy(space, start_x, start_y, finish_x, finish_y, state_value, (xloc, yloc, x_v, y_v)):
                # print('traject:', x, y)
                # print('greedy',greedy_policy(space, start_x, start_y, finish_x, finish_y, state_value, (xloc, yloc, x_v, y_v)))
                break
            ratio *= 1/9
    return state_value


if __name__ == '__main__':
    test = off_policy(100)
