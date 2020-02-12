[//]: # (Image References)


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The final goal is solving this environment with utilizing the Reinforcement learning models for continious actions. 

![gif](Images/reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

This project can be done with two different versions of the unity environment: 
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

In this project, the first version has been used. 

### Solving the Environment with the first version 

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.


### Set up the environment 

The environment can be downloaded from one of the links below (download link for all operating systems are available).

        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)

        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)

        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)

        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

### Instruction 
 
 The notebook ['continious_Control.py'](http://localhost:8888/notebooks/Continuous_Control.ipynb) includes the code that is required for setting up the environment. Deep Deterministic Policy Gradient has been used as a suggested solution. The agent, deep Q_Network and memory buffer are implemented in 'Agent_DDPG' as well as the actor and critic model which is written in 'model.py' file. 
 
 This  [paper](https://arxiv.org/pdf/1509.02971.pdf) was followed during the project. 

 After training, the model weights were saved in the two following files `checkpoint_actor.pth` and `checkpint_critic.pth`.
