import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_hidden=64, fc2_hidden=64): #initialize the parameters inorder to build a model
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed) #sets the seed for generating random numbers , returns a torch.Generator object
        self.fc1 = nn.Linear(state_size, fc1_hidden)
        self.fc2 = nn.Linear(fc1_hidden, fc2_hidden)
        self.fc3 = nn.Linear(fc2_hidden, action_size)

    def forward(self, state): #building a network that map the state to the action values
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
