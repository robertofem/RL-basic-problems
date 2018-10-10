# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque

GRID_WIDTH  = 12
GRID_HEIGHT = 4

class CliffWalkingEnv:
    def __init__(self):
        # Size of the space
        self.max_x = GRID_WIDTH
        self.max_y = GRID_HEIGHT
        # Possible actions
        self.actions = {0:"Up", 1:"Down", 2:"Left", 3:"Right"}

    def reset():
        #Reset the environment (start a new episode)
        self.X = 0
        self.Y = 0

    def step(action):
        #Move depending on the action
        if action == "Up":
            if self.Y < (self.max_y - 1):
                self.Y += 1
        else if action == "Down":
            if self.Y > 0:
                self.Y -= 1
        else if action == "Right":
            if self.X < (self.max_x - 1):
                self.X += 1
        else if action == "Left":
            if self.X > 0
                self.X -= 1
        state =  = np.matrix(X,Y)

        #Calculate reward
        if self._inside_cliff(self.X, self.Y):
            reward = -100
            X = 0
            Y = 0
        else
            reward = -1

        #Calculate done
        if (self.X == (self.max_x - 1)) and (self.Y == 0):
            done = 1
        else:
            done = 0

        return state, reward, done, None

    def _inside_cliff(self, X, Y):
        if (Y == 0) and (X > 0) and (X < (self.max_x - 1)):
            return True
        else:
            return False

class Agent:
    def __init__(self):
        _build_model()

    def _build_model(self):
        #Create the agent (a table)
        self.reset_model()

    def reset_model(self):
        # Create the model all with zeros
        self.model = np.zeros (GRID_WIDTH, GRID_HEIGHT, 4)
        # Initialize random Q table (except the terminal state that is 0)
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                for action in range(4):
                    if not ((Y == 0) and (X == (self.max_x - 1))):
                        self.model[x, y, action] = np.random.rand()*100

        return

    def train(self):



    def reset
