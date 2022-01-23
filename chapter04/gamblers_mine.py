import numpy as np
import matplotlib.pyplot as plt

discount_rate = 1
reward = 0
states = np.arange(100+1, dtype='int')
state_values = np.zeros_like(states, dtype='float')
state_values[-1] = 1
policy = np.zeros_like(states)
win_prob = 0.4
diff=10
iterations = 0
while diff > 1e-9:
    old_state = state_values.copy()
    for state in states[1:-1]:
        upper_action = min(state, 100 - state)
        action_values = []
        for action in range(1, upper_action+1):
            action_value = win_prob * (reward + discount_rate * state_values[state+action]) + (1-win_prob) * (reward + discount_rate * state_values[state-action])
            action_values.append(action_value)
        action = np.argmax(np.round(action_values,5))+1
        policy[state] = action
        state_values[state] = max(action_values)
        diff = (state_values - old_state).max()
    iterations += 1
    print(iterations)


