import numpy as np


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


def random_speed_change():
    vertical = np.random.randint(-1,2)
    horizontal = np.random.randint(-1,2)
    return vertical, horizontal


def is_finish(x, y, end_xs, end_ys):
    if x in end_xs and y in end_ys:
        return True
    else:
        return False


def in_road(x, y, space):
    if abs(x) >= space.shape[0] or abs(y) > space.shape[1]:
        return False
    if space[x, y]:
        return True
    else:
        return False


def start_init(start_xs, start_ys):
    xloc = np.random.randint(start_xs[0], start_xs[-1]+1)
    yloc = np.random.randint(start_ys[0], start_ys[-1]+1)
    velocity = [0,0]
    return xloc, yloc, velocity


def single_episode(space, start_xs, start_ys, end_xs, end_ys, epsilon=0.1):
    xloc, yloc, velocity = start_init(start_xs, start_ys)
    reward = 0
    trajectory = []
    while not is_finish(xloc, yloc, end_xs, end_ys):
        if np.random.random() <= epsilon:
            x_delta, y_delta = 0,0
        else:
            x_delta, y_delta = random_speed_change()
        velocity[0] = max(min(5, velocity[0]+x_delta), 0)
        velocity[1] = max(min(5, velocity[1]+y_delta), 0)
        reward += -1
        xloc -= velocity[0]
        yloc += velocity[1]
        trajectory.append((xloc, yloc, x_delta, y_delta))
        if not in_road(xloc, yloc, space):
            xloc, yloc, velocity = start_init(start_xs, start_ys)
    return reward, trajectory


if __name__ == '__main__':
    space, start_x, start_y, finish_x, finish_y = space_1()
    reward, trajectory = single_episode(space, start_x, start_y, finish_x, finish_y)
