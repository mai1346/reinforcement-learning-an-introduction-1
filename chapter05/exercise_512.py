import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import skimage.draw as skd
np.random.seed(5)


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

    start_x = np.full(6, fill_value=31, dtype='int')
    start_y = np.arange(3,9)
    finish_x = np.arange(6)
    finish_y = np.full_like(finish_x, fill_value=16)
    return space, start_x, start_y, finish_x,finish_y


def random_policy():
    vertical = np.random.randint(-1,2)
    horizontal = np.random.randint(-1,2)
    return vertical, horizontal


def greedy_policy(space, start_xs, start_ys, finish_line, state_values, state):
    actions_values = np.zeros((3,3))
    for col,h in enumerate(range(-1,2)):
        for row, v in enumerate(range(-1,2)):
            new_horizontal_vel = max(min(4, state[3] + v), 0)
            new_vertical_vel = max(min(4, state[2] + h), 0)
            new_x = state[0] - new_vertical_vel
            new_y = state[1] + new_horizontal_vel
            finished, point = is_finish(*state[:2], new_x, new_y, finish_line)
            if in_road(new_x, new_y, space):
                new_state_value = state_values[new_x, new_y, new_vertical_vel, new_horizontal_vel]
            elif finished:
                # 如果finish，则返回与重点线交点的state value.
                new_state_value = state_values[point[0], point[1], new_vertical_vel, new_horizontal_vel]
            else:
                sx, sy, _ = start_init(start_xs, start_ys)
                new_state_value = state_values[sx, sy, 0, 0]
            # new_state_value = state_values[new_x, new_y, new_horizontal_vel, new_vertical_vel]
            actions_values[row, col] = new_state_value
    # print(actions_values)
    index = np.unravel_index(np.argmax(actions_values),(3,3))
    return index[0]-1, index[1]-1


def is_finish(x, y, new_x, new_y, finish_line):
    """
    v1:是否越过终点线，并不是停留在终点线上，而是应该x属于end_xs,而y>= end_ys;
    v2:是否越过终点线，要看当前位置下，速度的向量是否与终点向量相交。
    @param x:
    @param y:
    @param end_xs:
    @param end_ys:
    @return:
    """
    projected_line = np.array(skd.draw.line(x, y, new_x, new_y)).T
    projected_line = [tuple(x) for x in projected_line]
    for point in projected_line:
        if point in finish_line:
            return True, point
    else:
        return False, None


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


def make_trajectory(space, start_xs, start_ys, finish_line, policy, epsilon=0.1):
    xloc, yloc, velocity = start_init(start_xs, start_ys)
    trajectory = []
    while True:
        # print(velocity)
        if np.random.random() < epsilon:
            vertical_delta, horizontal_delta = 0,0
        else:
            action_idx = np.random.choice(range(9), p=policy[xloc, yloc, velocity[0], velocity[1],:])
            act_mul_idx = np.unravel_index(action_idx, (3,3))
            vertical_delta, horizontal_delta = np.array(act_mul_idx) - 1
        trajectory.append((xloc, yloc, velocity[0], velocity[1], vertical_delta, horizontal_delta))
        # 注意 横向和纵向的坐标与np的二维数组的位置区别
        velocity[0] = max(min(4, velocity[0]+vertical_delta), 0)
        velocity[1] = max(min(4, velocity[1]+horizontal_delta), 0)
        if velocity==[0,0]:
            velocity = [1, 0]
        new_xloc = xloc - velocity[0]
        new_yloc = yloc + velocity[1]
        if is_finish(xloc, yloc, new_xloc, new_yloc, finish_line)[0]:
            break
        if not in_road(new_xloc, new_yloc, space):
            xloc, yloc, velocity = start_init(start_xs, start_ys)
        else:
            xloc = new_xloc
            yloc = new_yloc
    return np.array(trajectory)


def off_policy(episodes, eps=0):
    """
    every_visit
    @param episodes:
    @return:
    """
    space, start_x, start_y, finish_x, finish_y = space_1()
    finish_line = np.array([finish_x, finish_y]).T
    finish_line = [tuple(x) for x in finish_line]
    state_value = np.random.randn(space.shape[0], space.shape[1], 5, 5, 9) *400 -500
    cum_ratio = np.zeros_like(state_value, dtype='float')
    policy = np.ones_like(state_value, dtype='float') / 9
    for i in trange(episodes):
        # print(i)
        trajectory = make_trajectory(space, start_x, start_y, finish_line, policy, epsilon=eps)
        # print(i, len(trajectory))
        reward = 0
        ratio = 1
        for (xloc, yloc, x_v, y_v, x, y) in trajectory[::-1]:
            # print(reward)
            st_idx = (xloc, yloc, x_v, y_v)
            action_idx = np.ravel_multi_index((x+1,y+1),(3,3))
            sa_idx = st_idx+(action_idx,)
            reward += -1
            cum_ratio[sa_idx] += ratio
            state_value[sa_idx] += ratio * (reward - state_value[sa_idx]) / cum_ratio[sa_idx]
            greedy_action_ridxs = np.argmax(state_value[st_idx])
            policy[st_idx] = eps / 9
            policy[st_idx+(greedy_action_ridxs,)] = 1 - eps + eps / 9
            if action_idx != greedy_action_ridxs:
                # print('traject:', x, y)
                # print('greedy',greedy_policy(space, start_x, start_y, finish_x, finish_y, state_value, (xloc, yloc, x_v, y_v)))
                break
            ratio *= policy[sa_idx]
    return state_value, policy


if __name__ == '__main__':
    space, start_x, start_y, finish_x, finish_y = space_1()
    finish_line = np.array([finish_x, finish_y]).T
    finish_line = [tuple(x) for x in finish_line]
    test = off_policy(100000, eps=0.1)
