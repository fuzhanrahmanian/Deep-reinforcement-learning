# Report
---
This project utilised the DDPG (Deep Deterministic Policy Gradient) algorithm.

## State and Action Spaces
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.

## Learning Algorithm

The agent training utilised the `ddpg` function in the Continuous_Control notebook.

It continues episodical training via the the ddpg agent until `n_episoses` is at least 100 episodes and reached to the score (average reward) above +30. Note if the number of agents is >1 then the average reward of all agents at that step is used.

Each episode continues until `max_timesteps` time-steps is reached or until the environment says it's done.

As above, a reward of +0.1 is provided for each step that the agent's hand is in the goal location.

The DDPG agent is contained in `Agent_DDPG.py`

For each time step and agent the Agent acts upon the state utilising a shared (at class level) `replay_buffer`, `actor_local`, `actor_target`, `actor_optimizer`, `critic_local`, `criticl_target` and `critic_optimizer` networks.

### DDPG Hyper Parameters
- n_episodes (int): maximum number of training episodes
- max_timesteps (int): maximum number of timesteps per episode
- num_agents: number of agents in the environment

Where
`n_episodes=600`, `max_timesteps=10000`


### DDPG Agent Hyper Parameters

- BUFFER_SIZE (int): replay buffer size
- BATCH_SIZ (int): mini batch size
- GAMMA (float): discount factor
- TAU (float): for soft update of target parameters
- LR_ACTOR (float): learning rate for optimizer
- LR_CRITIC (float): learning rate for optimizer
- WEIGHT_DECAY (float): L2 weight decay

Where 
`BUFFER_SIZE = int(1e6)`, `BATCH_SIZE = 128`, `GAMMA = 0.99`, `TAU = 1e-3`, `LR_ACTOR = 1e-4`, `LR_CRITIC = 1e-4` and `WEIGHT_DECAY = 0.0`


### Neural Networks

Actor and Critic network models were defined in `Model.py`. 

The Actor networks utilised two fully connected layers both with 128 units with relu activation and tanh activation for the action space. The network has an initial dimension the same as the state size.

The Critic networks utilised two fully connected layers both with 128 units with relu activation. The critic network has  an initial dimension the size of the state size plus action size.

### Rewards plot

![Plot](images/plot.png)

![Episode](images/episode.png)



### Future ideas

Several ideas can be suggested for future work;

- Using more layers for batch normalization 

- Using dropout on the agent. 

- Exploring other methods including Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) methods. 


