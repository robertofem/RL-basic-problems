# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

GRID_WIDTH  = 12
GRID_HEIGHT = 4

EPISODES = 500

class CliffWalkingEnv:
    def __init__(self):
        # Size of the space
        self.max_x = GRID_WIDTH
        self.max_y = GRID_HEIGHT
        # Possible actions
        self.ACTIONS = ["Up", "Down", "Left", "Right"]

    def reset(self):
        #Reset the environment (start a new episode)
        self.X = 0
        self.Y = 0
        self.state = np.matrix([self.X, self.Y])
        return self.state

    def step(self, action):
        #Move depending on the action
        action = self.ACTIONS[action]
        if action == "Up":
            if self.Y < (self.max_y - 1):
                self.Y += 1
        elif action == "Down":
            if self.Y > 0:
                self.Y -= 1
        elif action == "Right":
            if self.X < (self.max_x - 1):
                self.X += 1
        elif action == "Left":
            if self.X > 0:
                self.X -= 1

        #Calculate reward and cliff movement
        if self._inside_cliff(self.X, self.Y):
            reward = -100
            self.X = 0
            self.Y = 0
        else:
            reward = -1
        self.state = np.matrix([self.X, self.Y])

        #Calculate done
        if (self.X == (self.max_x - 1)) and (self.Y == 0):
            done = 1
        else:
            done = 0

        return self.state, reward, done, None

    def _inside_cliff(self, X, Y):
        if (Y == 0) and (X > 0) and (X < (self.max_x - 1)):
            return True
        else:
            return False

class Agent:
    def __init__(self, agent_type = "SARSA"):
        self.agent_type = agent_type
        self._build_model()
        # Define some constants for the learning
        self.EPSILON_DECAY = 0.95
        self.EPSILON_MIN = 0.0
        self.ALFA = 0.04 # learning rate
        self.GANMA = 0.95 # disccount factor
        # Reset the training variables
        self.epsilon = 1.0

    def _build_model(self):
        # Create the model all with zeros
        self.model = np.zeros([GRID_WIDTH, GRID_HEIGHT, 4])
        # Initialize random Q table (except the terminal state that is 0)
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                for action in range(4):
                    if not ((y == 0) and (x == (GRID_WIDTH - 1))):
                        self.model[x, y, action] = -12#np.random.rand()*-100

    def _choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(4)
        else:
            action = np.argmax(self.predict(state))
        return action

    def train_episode(self, env):
        if self.agent_type == "SARSA":
            return self._train_sarsa(env)
        if self.agent_type == "Q-Learning":
            return self._train_qlearning(env)
        if self.agent_type == "Expected SARSA":
            return self._train_expected_sarsa(env)
        if self.agent_type == "n-step SARSA":
            return self._train_nstep_sarsa(env)

    def _train_sarsa(self, env):
        state = env.reset()
        action = self._choose_action(state)
        done = False
        episode_reward = 0
        while not done:
            new_state, reward, done, _ = env.step(action)
            new_action = self._choose_action(new_state)
            # Q(S;A)<-Q(S;A) + alfa[R + ganma*Q(S';A') - Q(S;A)]
            self.model[state[0,0], state[0,1], action] \
                += self.ALFA* \
                (reward + self.GANMA*self.predict(new_state)[new_action] \
                - self.predict(state)[action])
            state = new_state
            action = new_action
            episode_reward += reward
        self.epsilon *= self.EPSILON_DECAY
        if self.epsilon < self.EPSILON_MIN:
            self.epsilon = self.EPSILON_MIN
        return episode_reward, self.epsilon

    def _train_qlearning(self, env):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = self._choose_action(state)
            new_state, reward, done, _ = env.step(action)
            # Q(S;A)<-Q(S;A) + alfa[R + ganma*maxQ(S';a) - Q(S;A)]
            self.model[state[0,0], state[0,1], action] \
                += self.ALFA* \
                (reward + self.GANMA*np.amax(self.predict(new_state)) \
                - self.predict(state)[action])
            state = new_state
            episode_reward += reward
        self.epsilon *= self.EPSILON_DECAY
        if self.epsilon < self.EPSILON_MIN:
            self.epsilon = self.EPSILON_MIN
        return episode_reward, self.epsilon

    def _train_expected_sarsa(self, env):
        state = env.reset()
        action = self._choose_action(state)
        done = False
        episode_reward = 0
        while not done:
            new_state, reward, done, _ = env.step(action)
            new_action = self._choose_action(new_state)
            # Q(S;A)<-Q(S;A) + alfa[R + E[Q(S';A')|S'] - Q(S;A)]
            self.model[state[0,0], state[0,1], action] \
                += self.ALFA* \
                (reward + self.GANMA*(1/4)*np.sum(self.predict(new_state)) \
                - self.predict(state)[action])
            state = new_state
            action = new_action
            episode_reward += reward
        self.epsilon *= self.EPSILON_DECAY
        if self.epsilon < self.EPSILON_MIN:
            self.epsilon = self.EPSILON_MIN
        return episode_reward, self.epsilon

    def _train_nstep_sarsa(self, env):
        N=3
        R = deque()
        A = deque()
        S = deque()
        state = env.reset()
        action = self._choose_action(state)
        done = False
        episode_reward = 0
        S.append(state)
        A.append(action)
        T = float("inf")
        t=0
        while not done:
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            R.append(reward)
            S.append(new_state)
            if done:#if St+1 terminal
                T = t + 1;
            else:
                action = self._choose_action(new_state)
                A.append(action)

            tau = t - N + 1
            if tau >= 0:
                G = 0.0
                for i in range(tau+1, min(tau+N,T)):
                    G += (self.GANMA**(i-tau-1)) * R[i]
                if (tau + N < T):
                    G = G + (self.GANMA**N)*self.predict(S[tau+N])[A[tau+N]]
                    # Q(S;A)<-Q(S;A) + alfa[R + ganma*Q(S';A') - Q(S;A)]
                    self.model[S[tau][0,0], S[tau][0,1], A[tau]] \
                    += self.ALFA* (G - self.predict(S[tau])[A[tau]])

            if tau == T - 1:
              break
            # Count the time
            t += 1

        self.epsilon *= self.EPSILON_DECAY
        if self.epsilon < self.EPSILON_MIN:
            self.epsilon = self.EPSILON_MIN
        return episode_reward, self.epsilon

    def predict(self, state):
        ret_val = self.model[state[0, 0], state[0, 1], :]
        return ret_val

if __name__ == "__main__":
    agent_types = ["SARSA","Q-Learning","Expected SARSA", "n-step SARSA"]
    #agent_types = ["n-step SARSA"]

    # Train
    rewards = {}
    rewards_average = {}
    for i in range(len(agent_types)):
        env = CliffWalkingEnv()
        agent = Agent(agent_types[i])
        rewards[i] = np.zeros([EPISODES])
        rewards_average[i] = np.zeros([EPISODES])
        for e in range(EPISODES):
            episode_reward, epsilon = agent.train_episode(env)
            rewards[i][e] = episode_reward
            rewards_average[i][e] = np.mean(rewards[i][max(0,e-20):e])
            print("episode: {} epsilon:{} reward:{} averaged reward:{}".format(e, epsilon, rewards[i][e], rewards_average[i][e]))

    #Plot
    fig, ax = plt.subplots()
    fig.suptitle('Rewards')
    for j in range(len(rewards_average)):
        ax.plot(range(len(rewards_average[j])), rewards_average[j], label=agent_types[j])
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
