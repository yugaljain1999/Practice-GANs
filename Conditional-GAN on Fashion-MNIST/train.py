import torch
from torch import nn
import torchvision
from torchvision import datasets
from model import Generator,Discriminator
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
train_mnist = datasets.FashionMNIST(root='.',train=True,download=True,transform=transform)

train_loader = DataLoader(train_mnist,batch_size=32,shuffle=True)

train_x,train_y = next(iter(train_loader))

epochs = 50
batch_size = 32
lr = 3e-4
generator_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr)

loss_function = nn.BCELoss()

for epoch in range(epochs):
    for idx,(real_samples,digit_labels) in enumerate(train_loader):
        real_samples = real_samples.to(device)
        digit_labels = digit_labels.to(device) # [batch_size , labels_size]
        fake_labels = torch.randint(0,10,(batch_size,)).to(device)  # Random integers - 1 to 10 of size (batch_size,1)- in prior method  fake labels was just zeros
        true_labels = torch.ones((batch_size,1)).to(device)
        latent_samples = torch.randn((batch_size,100)).to(device) # here dimensions of latent data is equal to 100
        generated_data = generator(latent_samples,fake_labels)


        # Discriminator
        # Here we will calculate two types of losses for discriminator i.e. true discriminator loss and fake discriminator loss
        # True discriminator loss tries to compare output of discriminator on true data and true labels(torch.ones((batch_size,)))
        # Fake discriminator loss tries to compare output of discriminator on generated data and fake labels(torch.zeros((batch_size,)))
        # Take average of both types of losses and backpropagate losses and optimize it.
        



        discriminator.zero_grad()
        discriminator_output_on_true_data = discriminator(real_samples.view(batch_size,784),digit_labels)
        true_discriminator_loss = loss_function(discriminator_output_on_true_data,true_labels)
        discriminator_output_on_generated_data = discriminator(generated_data.detach(),fake_labels).view(batch_size)
        discriminator_loss_true_data = loss_function(discriminator_output_on_true_data,true_labels)
        discriminator_loss_generated_data = loss_function(discriminator_output_on_generated_data,torch.zeros((batch_size,)).to(device))
        total_discriminator_loss = (discriminator_loss_true_data + discriminator_loss_generated_data)/2
        total_discriminator_loss.backward()
        discriminator_optimizer.step()
        
        # Generator
        # Generator loss is calculated for discriminator output on generated data and true labels
        
        generator.zero_grad()
        discriminator_output_generated_data = discriminator(generated_data,fake_labels)
        generator_loss = loss_function(discriminator_output_generated_data,true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Inference for random noise data
        if epoch%2==0 and idx == batch_size-1:
            #with torch.no_grad():
            #    noise = torch.randn((batch_size,100)).to(device)
            #    generated_data = generator(noise,fake_labels)
            #    for x in generated_data:
            #        plt.imshow(x.cpu().detach().view(28,28),cmap='gray_r')
            #        plt.show()
            #        break

            print('Epoch:{}'.format(epoch),'Loss_G:{}'.format(generator_loss))
            print('Epoch:{}'.format(epoch),'Loss_D:{}'.format(total_discriminator_loss))

        
