# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 5000

GANMA = 0.99    # discount rate
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
LEARNING_RATE = 0.0005
N = 10

#TRAIN = 1 trains to a file. TRAIN = 0 plays with the coefficients of the file.
TRAIN = 1;
FILE_NAME = "ann-weights.h5"

if __name__ == "__main__":
  env = gym.make('CartPole-v1')
  state_size = 2
  action_size = env.action_space.n

  #Create the artificial neural network (ann)
  ann = Sequential()

  ann.add(Dense(24, input_dim = state_size, activation='relu'))
  ann.add(Dense(24, activation='relu'))
  ann.add(Dense(2, activation='linear'))
  ann.compile(loss='mse', optimizer=Adam(LEARNING_RATE), metrics=['mae'])

  if TRAIN == 1:
    epsilon = EPSILON
    for e in range(EPISODES):
      R = deque()
      A = deque()
      S = deque()
      #Initial state and action
      state = env.reset()
      state = np.reshape(state[2:4], [1, state_size])
      epsilon *= EPSILON_DECAY;
      done = False
      t = 0
      #choose greedy action
      action = np.argmax(ann.predict(state)[0])
      S.append(state)
      A.append(action)
      T = float("inf")
      while not done:
        # env.render()
        # Take that action and see the next state and reward
        new_state, reward, done, _ = env.step(action)
        new_state = np.reshape(new_state[2:4], [1, state_size])
        R.append(reward)
        S.append(new_state)
        if done:#if St+1 terminal
          T = t + 1;
        else:
          # Select an new action (epsilon-greedy)
          if np.random.random() < epsilon:
            action = random.randrange(action_size)
          else:
            action = np.argmax(ann.predict(new_state)[0])
          A.append(action)

        tau = t - N + 1
        if tau >= 0:
          G = 0.0
          for i in range(tau+1, min(tau+N,T)):
            G += (GANMA**(i-tau-1)) * R[i]
          if (tau + N < T):
            G = G + (GANMA**N)*ann.predict(S[tau+N])[0][A[tau+N]]
            G_vec = ann.predict(S[tau])
            G_vec[0][A[tau]] = G
            ann.fit(S[tau], G_vec, epochs=1, verbose=0)

        if tau == T - 1:
          break
        # Count the time
        t += 1

      #print episode results
      print("episode: {}/{}, score: {}, e: {:.2}"
          .format(e, EPISODES, t, epsilon))

  else: #TRAIN = 0 (therefore just play some random plays)
    #agent.load("./save/cartpole-dqn.h5")
    pass
