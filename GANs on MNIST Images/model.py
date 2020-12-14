import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(784,1024), # input image is of size (28,28) so flatten it into (784,1) and transform image into 1024 features
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(1024,512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512,256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256,1), # here output layer has just 1 neuron to compute probability whether generated image is fake or real
                        nn.Sigmoid()
        )

    def forward(self,x):
        x = x.view(x.size(0),784)
        discriminator_output = self.model(x)
        return discriminator_output

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        # Start from dimensions of random data and end with same input image size
        self.model = nn.Sequential(
                        nn.Linear(100,256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256,512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512,1024),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(1024,784), # output features size is same as size of input image i.e. 784
                        nn.Tanh()
                    )
    def forward(self,x):
        generator_output = self.model(x)
        generator_output = generator_output.view(generator_output.size(0),1,28,28) # 784 -> (1,28,28) - In this scenario view function is to be use
        return generator_output




