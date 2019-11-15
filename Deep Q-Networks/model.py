import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_hidden = 64, fc2_hidden = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        #number of nodes in first hidden layer
        self.fc1 = nn.Linear(state_size, fc1_hidden)
        #number of nodes in the second layer 
        self.fc2 = nn.Linear(fc1_hidden, fc2_hidden) 
        #output , taking the action value 
        self.fc3 = nn.Linear(fc2_hidden, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        #forward pass 
        x = self.fc1(state)
        #pass from activation function 
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
