# Report
---
This project utilised the MADDPG (Multi Agent Deep Deterministic Policy Gradient) algorithm.

## An overview 
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Learning Algorithm

For training the agent the MADDPG algorithm has been implemented. This class uses two DDPG agents. ReplayBuffer is also used as a shared buffer between the agents. 

So, MADDPG will combine the states, actions, rewards, next-states and dones from both agents and will add them up to the Replay Buffer. 

The Actor network has 3 linear layers. In the first 2 layers, relu activation has been used and in the last layer, tanh has been utilized. 
Critic network has the same structure as actor. 

The environment has been solved in episode '1278'. The network weights of each agent has been saved in agent1_checkpoint_actor.pth,agent1_checkpoint_critic.pth, agent2_checkpoint_actor.pth and  agent2_checkpoint_critic.pth

### HyperParameters

BUFFER_SIZE = int(1e5)  # replay buffer size

BATCH_SIZE = 250         # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor

LR_CRITIC = 1e-3        # learning rate of the critic 

WEIGHT_DECAY = 0        # L2 weight decay

mu=0
 
theta=0.15
 
sigma=0.2

n_episodes=8000



### Rewards plot

The environment has been solved in episode 1278 with an  Average score of 0.501. 

![Plot](Images/plot.png)

![Episode](Images/score.png)



### Future ideas

Several ideas can be suggested for future work;

- Change the hyperparameters more systematically amd investigate about their effect while training.  

- Using dropout on the agent. 

- Exploring other methods including Multi Agent PPO (Proximal Policy Optimization) and multi agent DQN (Deep Q Network) 


