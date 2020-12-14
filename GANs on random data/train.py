import torch
from torch import nn
from model import Generator,Discriminator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Generate random data for training
import random
import math
train_len = 1024
batch_size = 32
training_data = torch.zeros((train_len,2))
training_data[:,0] = 2*math.pi*torch.rand([train_len]) # x = 2*pi*(random data)
training_data[:,1] = torch.sin(training_data[:,0])
train_labels = torch.zeros(train_len)
train_set = [(training_data[i],train_labels[i]) for i in range(train_len)] # set of training data and labels

train_loader = DataLoader(train_set,batch_size=32,shuffle=True)


epochs = 200
lr = 3e-4
loss_function = nn.BCELoss()

generator = Generator() 
discriminator = Discriminator()

# REMEMBER optimizers for Generator and Discriminator would be different as both are trying to optimize their individual weights

generator_optimizer = torch.optim.Adam(generator.parameters(),lr=lr) 
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr) 



for epoch in range(epochs):
    for idx,(real_samples,_) in enumerate(train_loader): # iterate over list of tuple

        real_samples_label = torch.ones((batch_size,1)) # here train_len specifies number of training examples
        latent_train_samples = torch.randn((batch_size,2)) # here number of dimensions of random latent data = 2
        generated_samples = generator(latent_train_samples)
        generate_samples_labels = torch.zeros((batch_size,1))
        # concatenate real samples and generated samples(generate from generator) and real samples labels and generated samples labels
        train_samples_set = torch.cat((real_samples,generated_samples))
        train_label_set = torch.cat((real_samples_label,generate_samples_labels))

        # optimize loss of discriminator
        discriminator.zero_grad()
        discriminator_output = discriminator(train_samples_set)
        loss_discriminator = loss_function(discriminator_output,train_label_set)
        loss_discriminator.backward()
        discriminator_optimizer.step()

        generator.zero_grad()
        generator_output = generator(latent_train_samples)
        generator_final_output = discriminator(generator_output)
        loss_generator = loss_function(generator_final_output,real_samples_label)
        loss_generator.backward()
        generator_optimizer.step()
        
        # plot generated samples during training 
        generated_output_ = generator_output.detach()
        plt.plot(generated_output_[:,0],generated_output_[:,1],'.')
        plt.show()
        

        if epoch%10==0 and idx == batch_size - 1: # After 10 epochs and when idx = batch_size-1
            print('Epoch:{}'.format(epoch),'Loss_G:{}'.format(loss_generator))
            print('Epoch:{}'.format(epoch),'Loss_D:{}'.format(loss_discriminator))
            






        
        
