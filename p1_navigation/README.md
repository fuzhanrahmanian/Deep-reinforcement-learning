[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation using Deep Reinforcement Learning

###About Deep Reinforcement Learning
 > Reinforcement learning is an important branch of machine
learning, where an agent learns to take actions that would yield
the most reward by interacting with the environment. Different
from supervised learning, reinforcement learning cannot learn
from samples provided by an experienced external supervisor.
Instead, it has to operate based on its own experience despite
that it faces with significant uncertainty about the environment. Reinforcement learning is defined not characterizing learning
methods, but by characterizing a learning problem. Any method
that is suitable for solving that problem can be considered as a
reinforcement learning method [(ref.)](https://link.springer.com/content/pdf/10.1023/A:1022620604568.pdf).

### About the Navigation Project

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]



A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.


### About the environment
The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents)

>  he Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, 
>or other machine learning methods through a simple-to-use Python API. 




### Getting Started
- Create a Python 3.6 or 3.7 environment with Pythorch 0.4.0
- Clone this project if it is not already available
- Install the gym modules in order to use Box2D. 
If the module does not compile correctly, download and install a pre-compiled version from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
- Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

- Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 



