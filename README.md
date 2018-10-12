# RL-basic-problems

## Introduction
Test of different Reinforcement Learning (RL) algorithms in basic problems
using Python 3, neural nets in Keras and Gym.

Problems:
  * Inverted Pendulum: Solve the inverted pendulum problem in Gym using Q-learning.


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
