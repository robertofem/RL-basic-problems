# RL-basic-problems

## Introduction
Test of different Reinforcement Learning (RL) algorithms in basic problems
using Python 3, neural nets in Keras and Gym.

Problems:
  * Inverted Pendulum: try to solve the Gym inverted pendulum problem using Q-learning for reinforcement learning and and a neural network as agent. It does not converge. It may be a programming error on the reinforcement learning algorithm or an error on how the neural network is trained (probably can be solved in the following way: for each time step retraining the network with a bunch of previous time steps randomly selected instead of just traning with the results of the last time step). (CHRISTIAN, USE THIS ONLY TO CHECK THE SYNTHAX FOR CALLING NEURAL NETWORKS)

  * Cliff walking: solves the the cliff walking problem (instructions inside the folder) with 4 different reinforcement learning algorithms using a table as agent. It converges properly and fast for all different reinforcement algorithms tested. The program plots the reward vs time step for the different algorithms tested.
 (CHRISTIAN, COPY FROM HERE THE AGENTS TO BE SURE THAT Q-LEARNING AND OTHER METHODS ARE PROPERLY PROGRAMMED)


## Install libraries Windows

First install python 3.6 64-bit (last version compatible with tensorflow). The following  libraries are required:
  * numpy
  * gym
  * tensorflow
  * keras

Install virtualenv using pip and create a virtual environment:
```bash
 $ pip3 install virtualenvwrapper-win
 $ mkvirtualenv rl
```

Install numpy and gym using:
```bash
 (rl) $ pip install numpy
 (rl) $ pip install gym
```

Install TensorFlow (it is needed to specify the location of the file) with
```bash
 (rl) $ python -m pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.11.0-cp36-cp36m-win_amd64.whl
```

Install Keras with:
```bash
#Install some dependencies
 (rl) $ pip install numpy scipy
 (rl) $ pip install scikit-learn
 (rl) $ pip install pillow
 (rl) $ pip install h5py
#Install keras with tensorflow backend
 (rl) pip install keras
```
