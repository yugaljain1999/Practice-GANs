import torch
from model import Generator,Discriminator
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

torch.manual_seed(111)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
train_set = datasets.MNIST(root='.',download=True,train=True,transform=transform)

train_loader = DataLoader(train_set,shuffle=True,batch_size=32)
train_x,train_y = next(iter(train_loader))
for _ in range(16):
    ax = plt.subplot(4,4,_+1)
    plt.imshow(train_x[_].reshape(28,28),cmap='gray_r')
    plt.xticks([])
    plt.yticks([])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# Initialize model 
generator = Generator().to(device)
discriminator = Discriminator().to(device)


############## TRAINING ###############

epochs = 50
batch_size = 32
lr = 3e-4
generator_optimizer = torch.optim.Adam(generator.parameters(),lr = lr)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr = lr)
loss_function = nn.BCELoss() # BCE is Binary Cross Entropy loss function for binary classification(real or fake)



### Remember one thing about GANS - This doesn't care much about labels associated with each image as end result is generation of data(which is actually fake but discriminator unable to classify it as fake due to generator)
for epoch in range(epochs):
    for idx,(real_samples,_) in enumerate(train_loader):
        real_samples = real_samples.to(device=device)
        latent_samples =  torch.randn((batch_size,100)).to(device=device) # Random data - batch_size = 32 and dimensions = 100 
        generator_output_samples = generator(latent_samples)
        real_samples_labels = torch.ones((batch_size,1)).to(device=device) # dimension of real samples labels is equal to 1
        generator_output_labels = torch.zeros((batch_size,1)).to(device=device)
        all_samples = torch.cat((real_samples,generator_output_samples))
        all_samples_labels = torch.cat((real_samples_labels,generator_output_labels)) 

        #### make gradients of discriminator zero and optimize it's loss function ###
        discriminator.zero_grad()
        discriminator_outputs = discriminator(all_samples).to(device)
        discriminator_loss = loss_function(discriminator_outputs,all_samples_labels)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator.zero_grad()
        generator_outputs = generator(latent_samples)
        discriminator_generator_outputs = discriminator(generator_outputs).to(device)
        generator_loss = loss_function(discriminator_generator_outputs,real_samples_labels)
        generator_loss.backward()
        generator_optimizer.step()


        if epoch%10==0 and idx == batch_size - 1:
            print('Epoch:{}'.format(epoch),'Loss_G:{}'.format(generator_loss))
            print('Epoch:{}'.format(epoch),'Loss_D:{}'.format(discriminator_loss))





       
