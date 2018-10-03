# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

GANMA = 0.95    # discount rate
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001

#TRAIN = 1 trains to a file. TRAIN = 0 plays with the coefficients of the file.
TRAIN = 1;
FILE_NAME = "ann-weights.h5"

if __name__ == "__main__":
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n

  #Create the artificial neural network (ann)
  ann = Sequential()

  ann.add(Dense(10, input_dim = state_size, activation='sigmoid'))
  ##ann.add(Dense(10, activation='sigmoid'))
  ann.add(Dense(2, activation='linear'))
  ann.compile(loss='mse', optimizer='adam', metrics=['mae'])

  if TRAIN == 1:
    epsilon = EPSILON
    for e in range(EPISODES):
      #Initial state and action
      state = env.reset()
      state = np.reshape(state, [1, state_size])
      epsilon *= EPSILON_DECAY;
      done = False
      time = 0
      while not done:
        env.render()
        # Select an action (epsilon-greedy)
        if np.random.random() < epsilon:
            action = random.randrange(action_size)
        else:
            act_values = ann.predict(state)
            action = np.argmax(act_values[0])
        # Take that action and see the next state and reward
        new_state, reward, done, _ = env.step(action)
        new_state = np.reshape(new_state, [1, state_size])
        # Train the model if the episode finished
        target = reward + GANMA * np.max(ann.predict(new_state))
        target_vec = ann.predict(state)
        target_vec[0][action] = target
        ann.fit(state, target_vec.reshape(-1, 2), epochs=1, verbose=0)
        state = new_state
        # Count the time
        time += 1


      #print episode results
      print("episode: {}/{}, score: {}, e: {:.2}"
          .format(e, EPISODES, time, epsilon))

  else: #TRAIN = 0 (therefore just play some random plays)
    #agent.load("./save/cartpole-dqn.h5")
    pass
