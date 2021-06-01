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


#%%
# 生成隨機table作為範例
value = np.zeros((8,3))

# 舉一個例子，取出V(T1)，也就是取出第三列第零行的元素
# v_t1 = value[3,0]

# parameter
prob_action1 = 0.5
prob_action2 = 1 - prob_action1
action1_transform_prob = [[0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0, 1], [0.2, 0.8]]
action2_transform_prob = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0, 1], [0.8, 0.2]]
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

s_value = 0 # state_value
q_value_a1 = 1 # Q-value-a1
q_value_a2 = 2 # Q-value-a2



# turn left or right, get their prob.
left = 0
right = 1

# (from, to) : reward
reward = {(S0 , S1) : 15, (S0, S2) : 7, (S1, T1) : 7, (S1, S3) : -15, (S2, S3) : -20, 
          (S2, S4) : 5, (S3, T2) : 30, (S4, T2) : 0, (S4, T3) : 10}




#%%

check_list = [S4, S3, S2, S1, S0]
temp = []
left_reward = 0
right_reward = 0
for key, value in reward.items():
    if key[0] == check_list[0]:
        temp.append(key)

    
left_reward = reward[(S4, T2)]
right_reward = reward[(S4, T3)]

# state value
left_value = prob_action1 * (action1_transform_prob[S4 -1][left] * (left_reward + discount_factor * value[T2, s_value]) +
                action1_transform_prob[S4 - 1][right] * (right_reward + discount_factor * value[T3, s_value]))

right_value = prob_action2 * (action2_transform_prob[S4 - 1][left] * (left_reward + discount_factor * value[T2, s_value]) +
                action2_transform_prob[S4 - 1][right] * (right_reward + discount_factor * value[T3, s_value]))

value[S4, s_value] = left_value + right_value


# q_value_a1
left_value = action1_transform_prob[S4 -1][left] * (left_reward + discount_factor * 
            (prob_action1 * value[T2, q_value_a1] + prob_action2 * value[T2, q_value_a2]))
 

right_value = action1_transform_prob[S4 -1][right] * (right_reward + discount_factor * 
            (prob_action1 * value[T3, q_value_a1] + prob_action2 * value[T3, q_value_a2]))

value[S4, q_value_a1] = left_value + right_value


# q_value_a2
left_value = action2_transform_prob[S4 -1][left] * (left_reward + discount_factor * 
            (prob_action1 * value[T2, q_value_a1] + prob_action2 * value[T2, q_value_a2]))
 

right_value = action2_transform_prob[S4 -1][right] * (right_reward + discount_factor * 
            (prob_action1 * value[T3, q_value_a1] + prob_action2 * value[T3, q_value_a2]))

value[S4, q_value_a2] = left_value + right_value

#%%



# 儲存根據(a)提供的policy計算出的value的array
np.save('value_a.npy',value)

