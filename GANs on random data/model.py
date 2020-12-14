import torch
from torch import nn
import numpy as np
import random

# Declare Descriminator 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__() # initialize class to import functions and layers from nn.Module  
        self.model = nn.Sequential(
            nn.Linear(2,256), # 256 is number of hidden neurons in first hidden layer of neural network
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,1),
            nn.Sigmoid() # to classify whether generated data is fake or real 
        )
    def forward(self,x):
        discriminator_output = self.model(x)
        return discriminator_output

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2,16), # Dimensions of random training data is 2
            nn.Linear(16,32),
            nn.Linear(32,64),
            nn.Linear(64,2) # ended with dimensions equal to 2
        )

    def forward(self,x):
        generated_output = self.model(x)
        return generated_output



        







        