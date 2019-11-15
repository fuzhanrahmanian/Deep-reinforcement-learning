# Udacity Deep Reinforcement Learning Nanodegree
## Project 1: Navigation
### Train an RL Agent to Collect Bananas

![navigation](https://user-images.githubusercontent.com/52014627/68973702-340bfb00-07ef-11ea-8e13-d238d007fa25.gif)
*Agent in action*

### Goal
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 
is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. In order to solve the environment, the agent must achieve an average score of +13 over 100 consecutive episodes.
This is achieved by using a Unity environment adopted by Udacity available for
[Windows 64-Bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip).
 

 ### Approach
 
 #### 1. Start the Environment and import necessary packages
 Basic starting procedure for the initialization of the environment. 
 
 #### 2. Examine the State and Action Spaces
 The environment has a dimension of 37, inside which the action can take space. 
 These are the possible discrete actions:
 
 - `0` move forward
- `1` move backward
- `2` turn left
- `3` turn right


 #### 3. Take Random Actions in the Environment
 Below, the agent will select a random action (uniformly), testing it for each time step 

```python
env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))
```
 
 #### 4. Training the agent with Deep Q_Network
 The learning algorithm used here is Deep Q_Networks referenced in 
 [this paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).
 In the state input vector a deep neural network is used as listed in the following:
 
- Fully connected layer - input: 37 (state size) output: 64
- Fully connected layer - input: 64 output 64
- Fully connected layer - input: 64 output: (action size)

The DQN algorithm was implemented with the following parameters:

- Maximum steps per episode: 1000
- Starting epsilon: 1.0
- Ending epsilon: 0.01
- Epsilon decay rate: 0.999
- Number of episodes: 2200

In every step, the agent will select an action based on the epsilon greedy policy. 
If the random selected probability is smaller than the epsilon,
 the agent will choose an equiprobable policy (non greedy action and greedy action). 
 Else the agent will choose the greedy policy and will take the maximum q-value function.
 ###### &nbsp;
 In order to break the correlation between the experienced tuples and let the agent 
 learn more about the past experiences, instead of just focusing on the future steps, 
 the replay buffer is used.
 
  
 
 ### Results
 
![image](https://user-images.githubusercontent.com/52014627/68976454-142c0580-07f6-11ea-97d4-861ca130b00a.png)

 ###### &nbsp;
This plots shows the rewards gained by the agent during training and at episode 1875 the 
mean-score of the agent reached and surpassed 13.

![image](https://user-images.githubusercontent.com/52014627/68976537-42114a00-07f6-11ea-9eb8-2829532b8321.png)

### Improvements
- Double DQNs: in order to stop the algorithm from incidental high rewards that may have been obtained by chance.
This will stop the DQN from overestimating the Q-Value function. 
- Prioritized experience reply for learning more efficiently from some transitions this algorithm can be applied
which means that the important transition should be sampled with higher probability. 
- Duelling DQNs: since estimating the action of some states does not make difference, this method is presented in order 
to access the value of each state without having to learn the effect of each action and just estimate the value
of the actions which are prioritized. 