# 问题的模型化
# 1. action是什么？ 本问题中action是在AB两地之前运送的车辆数量。
# 2. 所有可能的action是什么？ 由于限制最多5辆车，对于A地来说，所有可能的action是-5到+5之间的所有整数。
# 3. state是什么？state是AB两地每天最终的车辆数量。由于每地最多只能有20辆车，所以一个完整的包含所有state的矩阵形状为（20+1，20+1）
# 4. 如何计算state value？对于一个给定的policy，某个state的value 是其每个action value的期望，这里我们的基准策略就是永远移动0辆车，也就是p(action=0)=1,其他的p=0。
# 5. policy improvement. 对于某个state，如果新的policy下的某个action state value 不低于当前的state value，我们就用这个policy替换当前的policy，遍历完所有的state之后，
# 对新的policy进行evaluate，直到没有新的policy不再发生变化，我们就认为达到了policy stable。
import datetime as dt
import numpy as np
from scipy.stats import poisson


def cartesian_prod(row_arr1, col_arr2):
    row_locs = np.repeat(row_arr1, len(col_arr2))
    col_locs = np.tile(col_arr2, len(row_arr1))
    return row_locs, col_locs


def expected_return(a_car_num, action, b_car_num, state_values):
    car_move_cost = cost_moving_per_car * abs(action)
    a_valid_rental_num = np.minimum(a_car_num - action, a_rental_num)
    a_car_loc = np.minimum(a_car_num - a_valid_rental_num + a_return_lambda - action, max_car_per_loc)
    a_state = np.repeat(a_car_loc, 11)
    b_valid_rental_num = np.minimum(b_car_num + action, b_rental_num)
    ab_rental_sum = a_valid_rental_num[:, np.newaxis] + b_valid_rental_num
    ab_rental_profit = ab_rental_sum * profit_per_car
    b_car_loc = np.minimum(b_car_num - b_valid_rental_num + b_return_lambda + action, max_car_per_loc)
    b_state = np.tile(b_car_loc, 11)
    seq_state_value = state_values[a_state, b_state].reshape(-1, 11)
    expect_ret = np.sum(ab_rental_prob_matrix * (ab_rental_profit + discount_rate * seq_state_value)) - car_move_cost
    return expect_ret


def policy_eval(policy, state_values):
    diff = 10
    # iterations = 0
    while diff > 1e-4:
        old = state_values.copy()
        for a_car_num in range(max_car_per_loc + 1):
            for b_car_num in range(max_car_per_loc + 1):
                # 在场给定state的前提下，计算100种场景的期望收益即为当下state 的value，期望收益包含租车回报，每台10。以及后续state的value折现。
                action = policy[a_car_num, b_car_num]
                # state为当天结束时的车辆，在晚上挪车，那么对应白天的可用车辆要去掉移走的车辆
                state_values[a_car_num, b_car_num] = expected_return(a_car_num, action, b_car_num, state_values)
        diff = abs(state_values - old).max()
        # iterations += 1
        # print(iterations)


def policy_impovement(actions, policy, state_values):
    stable = True
    for a_car_num in range(max_car_per_loc + 1):
        for b_car_num in range(max_car_per_loc + 1):
            old_action = policy[a_car_num, b_car_num]
            action_values = []
            for action in actions:
                if (0 <= action <= a_car_num) or (-b_car_num <= action <= 0):
                    action_value = expected_return(a_car_num,action, b_car_num, state_values)
                else:
                    action_value = -np.inf
                action_values.append(action_value)
            new_action = actions[np.argmax(action_values)]
            if new_action != old_action:
                stable = False
            policy[a_car_num, b_car_num] = new_action

    return stable


if __name__ == '__main__':
    start = dt.datetime.now()
    profit_per_car = 10
    cost_moving_per_car = 2
    max_car_per_loc = 20
    max_movable_car = 5

    # poisson distribution
    a_rental_lambda = 3
    a_return_lambda = 3
    b_rental_lambda = 4
    b_return_lambda = 2

    # bellman equation
    discount_rate = 0.9

    actions = np.arange(-max_movable_car, max_movable_car + 1)

    # 计算一个车都不挪的策略的state value，通过迭代收敛
    state_values = np.zeros((max_car_per_loc + 1, max_car_per_loc + 1))  # 行索引代表A的车数量，列索引代表B的车数量
    # 期望为3和4的泊松分布的概率质量函数在n=11时，poisson.pmf(11,4)=0.001924, poisson.pmf(11,3)=0.000220,属于小概率事件。
    # 所以我们假设ab两地车辆需求的值小于11，即取[0,10],计算每一个需求情况的概率, 合计又10*10=100种场景。
    a_rental_num = np.arange(11)
    b_rental_num = np.arange(11)
    # 只需要计算一次泊松分布的概率
    a_rental_probs = np.array([poisson.pmf(x, a_rental_lambda) for x in a_rental_num])
    b_rental_probs = np.array([poisson.pmf(x, b_rental_lambda) for x in b_rental_num])
    ab_rental_prob_matrix = a_rental_probs[:, np.newaxis] * b_rental_probs

    policy = np.zeros_like(state_values, dtype='int')
    iterations = 0
    while True:
        policy_eval(policy, state_values)
        stable = policy_impovement(actions, policy, state_values)
        iterations += 1
        print('iterations:', iterations)
        if stable:
            break
    print(dt.datetime.now() - start)