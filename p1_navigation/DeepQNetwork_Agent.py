import numpy as np
import random
from collections import namedtuple, deque #namedtuple is factory function for creating tuple subclasses with named fields and deque is list_like container with fast appends and pops on either end
from model_representation import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # size of replay buffer
BATCH_SIZE = 64    # number of minibatch sites
GAMMA = 0.99  # discount factor
TAU = 1e-3  # is used for soft update of target parameters
LR = 5e-4  # value of learning rate
UPDATE_EVERY = 5 # shows how frequent the network should be updated
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():   # define the interactions from environment and learn from them
    def __init__(self, state_size, action_size, seed):
        # here I initialized an agent object
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Initialiting the Q_Networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr= LR)
        # initializing the replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done): # for saving the experience in the replay memory
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE: # when the collected samples were enough, then a random subset can be taken and learning process can start
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0): # choose an action for the given state and based on current policy
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()  # turn of the dropout and take the highest performance from the network
        with torch.no_grad(): # shutoff all the gradients
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps: # using the epsilon greedy action
            return np.argmax(action_values.cpu().data.numpy()) # greedy action
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma): # update the value parameters by using the given batch of experience tuples
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) # geting the maximum predicted Q value for the next states from the target model and us0e detach function to extract all those possible values
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones)) # computing the Q target for the current state
        Q_expected = self.qnetwork_local(states).gather(1, actions) # expected Q value from the local model
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad() # minimize the loss and clear all the gradients of parameters of previous calculation
        loss.backward()
        self.optimizer.step() # update the weight
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) # update the target network

    def soft_update(self, local_model, target_model, tau): # soft update model parameters
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1 - tau)*target_param.data)


class ReplayBuffer: # defining fixed-size buffer to store experience tuples
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done): # add a new experience to memory
        e = self.experience(state,action, reward, next_state, done)
        self.memory.append(e)

    def sample(self): # randomly sample a batch of experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device) #np.vstack: stack arrays in sequence vertically (row wise)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self): # current size of internal memory
        return len(self.memory)
