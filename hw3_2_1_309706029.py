import numpy as np
'''
請大家在計算value和Q-value的時候請使用shape為(8,3)的array儲存
並且使用下方的格式做對應
state value Q-value_a1 Q-value_a2
S0    ...   ...        ...
S1    ...   ...        ...
S2    ...   ...        ...
T1    ...   ...        ...
S3    ...   ...        ...
S4    ...   ...        ...
T2    ...   ...        ...
T3    ...   ...        ...
'''


# %%
# 生成隨機table作為範例
# value = np.zeros((8,3))

# 舉一個例子，取出V(T1)，也就是取出第三列第零行的元素
# v_t1 = value[3,0]

# parameter
prob_action1 = 0.5
prob_action2 = 1 - prob_action1
action1_transform_prob = [[0.2, 0.8], [
    0.7, 0.3], [0.3, 0.7], [0, 1], [0.2, 0.8]]
action2_transform_prob = [[0.9, 0.1], [
    0.2, 0.8], [0.7, 0.3], [0, 1], [0.8, 0.2]]
discount_factor = 0.9

# index of each node
S0 = 0
S1 = 1
S2 = 2
T1 = 3
# S3, S4如果要去查action1, 2的表，index要減1
S3 = 4
S4 = 5
T2 = 6
T3 = 7
null_node = 8  # for S3

s_value = 0  # state_value
q_value_a1 = 1  # Q-value-a1
q_value_a2 = 2  # Q-value-a2


# turn left or right, get their prob.
left = 0
right = 1

# (from, to) : reward

reward = {(S0, S1): 15, (S0, S2): 7, (S1, T1): 7, (S1, S3): -15, (S2, S3): -20,
          (S2, S4): 5, (S3, null_node): 0, (S3, T2): 30, (S4, T2): 0, (S4, T3): 10}


# %%

def calculate_S_Q_table_a(probability):
    value = np.zeros((8, 3))

    check_list = [S4, S3, S2, S1, S0]

    each_agent_prob = []  # [S4, S3, S2, S1, S0]
    for prob in probability:
        each_agent_prob.append([prob, 1-prob])

    for i in range(len(check_list)):
        temp = []

        prob_action1 = each_agent_prob[i][0]
        prob_action2 = each_agent_prob[i][1]

        for key, val in reward.items():
            if key[0] == check_list[i]:
                temp.append(key)

        from_node = temp[0][0]
        to1_node = temp[0][1]
        to2_node = temp[1][1]

        left_reward = reward[(from_node, to1_node)]
        right_reward = reward[(from_node, to2_node)]

        if from_node == S4:
            from_node -= 1

            # state value
            left_value = prob_action1 * (action1_transform_prob[from_node][left] * (left_reward + discount_factor * value[to1_node, s_value]) +
                                         action1_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            right_value = prob_action2 * (action2_transform_prob[from_node][left] * (left_reward + discount_factor * value[to1_node, s_value]) +
                                          action2_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            value[from_node + 1, s_value] = left_value + right_value

            # q_value_a1
            left_value = action1_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * value[to1_node, q_value_a1] + prob_action2 * value[to1_node, q_value_a2]))

            right_value = action1_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node + 1, q_value_a1] = left_value + right_value

            # q_value_a2
            left_value = action2_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * value[to1_node, q_value_a1] + prob_action2 * value[to1_node, q_value_a2]))

            right_value = action2_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node + 1, q_value_a2] = left_value + right_value

        elif from_node == S3:
            from_node -= 1

            # state value
            left_value = prob_action1 * (action1_transform_prob[from_node][left] * (left_reward + discount_factor * 0) +
                                         action1_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            right_value = prob_action2 * (action2_transform_prob[from_node][left] * (left_reward + discount_factor * 0) +
                                          action2_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            value[from_node + 1, s_value] = left_value + right_value

            # q_value_a1
            left_value = action1_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (0 * 0 + 0 * 0))

            right_value = action1_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node + 1, q_value_a1] = left_value + right_value

            # q_value_a2
            left_value = action2_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (0 * 0 + 0 * 0))

            right_value = action2_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node + 1, q_value_a2] = left_value + right_value

        else:
            # state value
            left_value = prob_action1 * (action1_transform_prob[from_node][left] * (left_reward + discount_factor * value[to1_node, s_value]) +
                                         action1_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            right_value = prob_action2 * (action2_transform_prob[from_node][left] * (left_reward + discount_factor * value[to1_node, s_value]) +
                                          action2_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            value[from_node, s_value] = left_value + right_value

            # q_value_a1
            left_value = action1_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * value[to1_node, q_value_a1] + prob_action2 * value[to1_node, q_value_a2]))

            right_value = action1_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node, q_value_a1] = left_value + right_value

            # q_value_a2
            left_value = action2_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * value[to1_node, q_value_a1] + prob_action2 * value[to1_node, q_value_a2]))

            right_value = action2_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node, q_value_a2] = left_value + right_value

    return value


uniform_policy = [0.5, 0.5, 0.5, 0.5, 0.5]
S_Q_table = calculate_S_Q_table_a(uniform_policy)

print('value_a', S_Q_table)
# 儲存根據(a)提供的policy計算出的value的array
np.save('value_a.npy', S_Q_table)

# %%


# left_reward = reward[(S4, T2)]
# right_reward = reward[(S4, T3)]

# # state value
# left_value = prob_action1 * (action1_transform_prob[S4 -1][left] * (left_reward + discount_factor * value[T2, s_value]) +
#                 action1_transform_prob[S4 - 1][right] * (right_reward + discount_factor * value[T3, s_value]))

# right_value = prob_action2 * (action2_transform_prob[S4 - 1][left] * (left_reward + discount_factor * value[T2, s_value]) +
#                 action2_transform_prob[S4 - 1][right] * (right_reward + discount_factor * value[T3, s_value]))

# value[S4, s_value] = left_value + right_value


# q_value_a1
# left_value = action1_transform_prob[S4 -1][left] * (left_reward + discount_factor *
#             (prob_action1 * value[T2, q_value_a1] + prob_action2 * value[T2, q_value_a2]))


# right_value = action1_transform_prob[S4 -1][right] * (right_reward + discount_factor *
#             (prob_action1 * value[T3, q_value_a1] + prob_action2 * value[T3, q_value_a2]))

# value[S4, q_value_a1] = left_value + right_value


# # q_value_a2
# left_value = action2_transform_prob[S4 -1][left] * (left_reward + discount_factor *
#             (prob_action1 * value[T2, q_value_a1] + prob_action2 * value[T2, q_value_a2]))


# right_value = action2_transform_prob[S4 -1][right] * (right_reward + discount_factor *
#             (prob_action1 * value[T3, q_value_a1] + prob_action2 * value[T3, q_value_a2]))

# value[S4, q_value_a2] = left_value + right_value

# %%

policy_Pi = [0, 0, 1, 0, 1]  # [S0, S1, S2, S3, S4]
# policy_Pi.reverse() # because check_list is [S4, S3, S2, S1, S0]


def calculate_S_Q_table_b(probability):
    value = np.zeros((8, 3))
    check_list = [S4, S3, S2, S1, S0]
    reverse_probability = probability[::-1]

    each_agent_prob = []  # [S4, S3, S2, S1, S0]
    for prob in reverse_probability:
        each_agent_prob.append([prob, 1-prob])

    for i in range(len(check_list)):
        temp = []

        prob_action1 = each_agent_prob[i][0]
        prob_action2 = each_agent_prob[i][1]

        for key, val in reward.items():
            if key[0] == check_list[i]:
                temp.append(key)

        from_node = temp[0][0]
        to1_node = temp[0][1]
        to2_node = temp[1][1]

        left_reward = reward[(from_node, to1_node)]
        right_reward = reward[(from_node, to2_node)]

        if from_node == S4:
            from_node -= 1

            # state value
            left_value = prob_action1 * (action1_transform_prob[from_node][left] * (left_reward + discount_factor * value[to1_node, s_value]) +
                                         action1_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            right_value = prob_action2 * (action2_transform_prob[from_node][left] * (left_reward + discount_factor * value[to1_node, s_value]) +
                                          action2_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            value[from_node + 1, s_value] = left_value + right_value

            # q_value_a1
            # because Pi(a' | s')的s'是terminal，所以再做下一個動作的選擇都為0, a2也是
            prob_action1 = 0
            prob_action2 = 0

            left_value = action1_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * value[to1_node, q_value_a1] + prob_action2 * value[to1_node, q_value_a2]))

            right_value = action1_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node + 1, q_value_a1] = left_value + right_value

            # q_value_a2

            left_value = action2_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * value[to1_node, q_value_a1] + prob_action2 * value[to1_node, q_value_a2]))

            right_value = action2_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node + 1, q_value_a2] = left_value + right_value

        elif from_node == S3:
            from_node -= 1

            # state value
            left_value = prob_action1 * (action1_transform_prob[from_node][left] * (left_reward + discount_factor * 0) +
                                         action1_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            right_value = prob_action2 * (action2_transform_prob[from_node][left] * (left_reward + discount_factor * 0) +
                                          action2_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            value[from_node + 1, s_value] = left_value + right_value

            # q_value_a1
            # because Pi(a' | s')的s'是terminal，所以再做下一個動作的選擇都為0, a2也是
            prob_action1 = 0
            prob_action2 = 0
            left_value = action1_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * 0 + prob_action1 * 0))

            right_value = action1_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node + 1, q_value_a1] = left_value + right_value

            # q_value_a2

            left_value = action2_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * 0 + prob_action1 * 0))

            right_value = action2_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node + 1, q_value_a2] = left_value + right_value

        else:
            # state value
            left_value = prob_action1 * (action1_transform_prob[from_node][left] * (left_reward + discount_factor * value[to1_node, s_value]) +
                                         action1_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            right_value = prob_action2 * (action2_transform_prob[from_node][left] * (left_reward + discount_factor * value[to1_node, s_value]) +
                                          action2_transform_prob[from_node][right] * (right_reward + discount_factor * value[to2_node, s_value]))

            value[from_node, s_value] = left_value + right_value

            # q_value_a1
            if to1_node >= S3:
                prob_action1 = probability[to1_node - 1]
                prob_action2 = 1-prob_action1
            else:
                prob_action1 = probability[to1_node]
                prob_action2 = 1-prob_action1
            left_value = action1_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * value[to1_node, q_value_a1] + prob_action2 * value[to1_node, q_value_a2]))

            if to2_node >= S3:
                prob_action1 = probability[to2_node - 1]
                prob_action2 = 1-prob_action1
            else:
                prob_action1 = probability[to2_node]
                prob_action2 = 1-prob_action1
            right_value = action1_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node, q_value_a1] = left_value + right_value

            # q_value_a2
            if to1_node >= S3:
                prob_action1 = probability[to1_node - 1]
                prob_action2 = 1-prob_action1
            else:
                prob_action1 = probability[to1_node]
                prob_action2 = 1-prob_action1
            left_value = action2_transform_prob[from_node][left] * (left_reward + discount_factor *
                                                                    (prob_action1 * value[to1_node, q_value_a1] + prob_action2 * value[to1_node, q_value_a2]))

            if to2_node >= S3:
                prob_action1 = probability[to2_node - 1]
                prob_action2 = 1-prob_action1
            else:
                prob_action1 = probability[to2_node]
                prob_action2 = 1-prob_action1
            right_value = action2_transform_prob[from_node][right] * (right_reward + discount_factor *
                                                                      (prob_action1 * value[to2_node, q_value_a1] + prob_action2 * value[to2_node, q_value_a2]))

            value[from_node, q_value_a2] = left_value + right_value

    return value


S_Q_table = calculate_S_Q_table_b(policy_Pi)

print('value_b', S_Q_table)
# 儲存根據(b)提供的policy計算出的value的array
np.save('value_b.npy', S_Q_table)


# %%

# # experiment
# policy_Pi = [0, 0, 1, 0, 1] # [S0, S1, S2, S3, S4]
# # q_value_a1
# prob_action1 = policy_Pi[S1]
# prob_action2 = 1-prob_action1
# left_reward = 15
# right_reward = 7
# value = S_Q_table

# left_value = action1_transform_prob[S0][left] * (left_reward + discount_factor *
#             (prob_action1 * value[S1, q_value_a1] + prob_action2 * value[S1, q_value_a2]))

# prob_action1 = policy_Pi[S2]
# prob_action2 = 1-prob_action1

# right_value = action1_transform_prob[S0][right] * (right_reward + discount_factor *
#             (prob_action1 * value[S2, q_value_a1] + prob_action2 * value[S2, q_value_a2]))

# S0_a1 = left_value + right_value
# print('S0_a1', S0_a1)

# # print(0.2 * (15 + 0.9 * (0 * 8.5 + 1 * 11)) +
# #       0.8 * (7 + 0.9 * (0 * 10.64 + 1 * 8.56)))

# # q_value_a2

# prob_action1 = policy_Pi[S3 - 1]
# prob_action2 = 1-prob_action1

# left_value = action2_transform_prob[S0][left] * (left_reward + discount_factor *
#             (prob_action1 * value[S1, q_value_a1] + prob_action2 * value[S1, q_value_a2]))

# prob_action1 = policy_Pi[S4 - 1]
# prob_action2 = 1-prob_action1

# right_value = action2_transform_prob[S0][right] * (right_reward + discount_factor *
#             (prob_action1 * value[S2, q_value_a1] + prob_action2 * value[S2, q_value_a2]))

# S0_a2 = left_value + right_value
# print('S0_a2', S0_a2)

# # print(0.9 * (15 + 0.9 * (0 * 8.5 + 1 * 11)) +
# #       0.1 * (7 + 0.9 * (0 * 10.64 + 1 * 8.56)))

# test_node = S4
# # policy_Pi = [0, 0, 1, 0, 1] # [S0, S1, S2, S3, S4]

# if test_node >= S3:
#     prob_action1 = policy_Pi[test_node - 1]
#     prob_action2 = 1-prob_action1
# else:
#     prob_action1 = policy_Pi[test_node]
#     prob_action2 = 1 - prob_action1


# print(prob_action1)
# print(prob_action2)
