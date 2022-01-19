import numpy as np

# actions
up = [0, 1]
down = [0, -1]
left = [-1, 0]
right = [1, 0]
# gridworld
hight = 4
width = 4
transition_reward = -1
# policy
prob = 0.25

state_values = np.zeros((hight, width))


def terminal(x_loc, y_loc):
    return (x_loc == 0 and y_loc == 0) or (x_loc == width - 1 and y_loc == hight - 1)


diff = 1e2
iter_count = 0
while diff > 1e-4:
    # result = np.zeros_like(state_values)
    old = state_values.copy()
    for h in range(hight):
        for w in range(width):
            if not terminal(w, h):
                state_value = transition_reward + prob * (
                            state_values[w, h - 1 if h > 0 else h] + state_values[w, h + 1 if h < hight - 1 else h] +
                            state_values[w - 1 if w > 0 else w, h] + state_values[w + 1 if w < width - 1 else w, h])
                state_values[h,w] = state_value
    diff = abs(state_values - old).mean()
    # state_values = result.copy()
    iter_count+=1

print(state_values)

