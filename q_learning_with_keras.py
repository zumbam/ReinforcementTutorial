
import gym
import numpy as np

def naive_sum_reward_agent(env, num_episodes=500):
    # this is the table that will hold our summated rewards for
    # each action in each state
    r_table = np.zeros((5, 2))
    for g in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.sum(r_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, 2)
            else:
                # select the action with highest cummulative reward
                a = np.argmax(r_table[s, :])
            new_s, r, done, _ = env.step(a)
            r_table[s, a] += r
            s = new_s
    return r_table

def q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.99
    lr = 0.8
    for i in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.sum(q_table[s,:]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, 2)
            else:
                # select the action with largest q value in state s
                a = np.argmax(q_table[s, :])
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr*(y*np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table


def eps_greedy_q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999

    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        i = 0
        while not done:

            # select the action with highest cummulative reward
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s, :])
            # pdb.set_trace()
            new_s, r, done, _ = env.step(a)
            i += 1
            q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
        print(f'done in {i}')
    return q_table

def run_game(table, env):
    s = env.reset()
    tot_reward = 0
    done = False
    while not done:
        a = np.argmax(table[s, :])
        s, r, done, _ = env.step(a)
        tot_reward += r
    return tot_reward

def test_methods(env, num_iterations=100):
    winner = np.zeros((3,))
    for g in range(num_iterations):
        m0_table = naive_sum_reward_agent(env, 500)
        m1_table = q_learning_with_table(env, 500)
        m2_table = eps_greedy_q_learning_with_table(env, 500)
        m0 = run_game(m0_table, env)
        m1 = run_game(m1_table, env)
        m2 = run_game(m2_table, env)
        w = np.argmax(np.array([m0, m1, m2]))
        winner[w] += 1
        print("Game {} of {}".format(g + 1, num_iterations))
    return winner

env = gym.make('NChain-v0')
# winner = eps_greedy_q_learning_with_table(env)
# print(winner)

from keras import Sequential
from keras.layers import InputLayer, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import os

timestamp = datetime.datetime.now().strftime('%d_%m_%y__%H_%M_%S')
model_name = 'model_' + timestamp
log_path = model_name + '/log'
checkpoint_path = model_name + '/model'
print(checkpoint_path)
# os.makedirs(log_path)
# os.makedirs(checkpoint_path)

# checkpoint_callback = ModelCheckpoint(checkpoint_path + '/weights.{epoch:02d}-{loss:.2f}.hdf5', save_best_only=True, monitor='loss')
# tensorboard_callback = TensorBoard(log_path, batch_size=1, update_freq=1)


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 5)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])



# # now execute the q learning
y = 0.95
eps = 0.5
decay_factor = 0.999
num_episodes = 1000
r_avg_list = []

for i in range(num_episodes):
    s = env.reset()
    eps *= decay_factor
    if i % 100 == 0:
        print("Episode {} of {}".format(i + 1, num_episodes))
    done = False
    r_sum = 0


    while not done:
        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
        new_s, r, done, _ = env.step(a)
        target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
        target_vec = model.predict(np.identity(5)[s:s + 1])[0]
        target_vec[a] = target
        model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
        s = new_s
        r_sum += r

    r_avg_list.append(r_sum / 1000)

r_avg_arr = np.array(r_avg_list)
np.savetxt('r_avg_list' + timestamp + '.txt', r_avg_arr)