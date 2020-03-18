import torch
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import warnings
from functools import wraps
import matplotlib.pyplot as plt

'''
Based on the template from: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/denoising-autoencoder/Denoising_Autoencoder_Solution.ipynb

Refrence: https://github.com/milindmalshe/Fully-Connected-Neural-Network-MNIST-Classification-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
Refrence: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

'''

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 128
# if we use dropout or not
dropout = False
# define the learning rate
learning_rate = 1e-3
# number of epochs to train the model
n_epochs = 1
# for adding noise to images
noise_factor = 0.5
# defines the size of the latent space
latent_space = 8
# weight decay for ADAM
weight_decay=1e-5

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# ignoring Pyorch warnings
def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

if torch.cuda.is_available():
    cuda = True
else:
    cuda = False

# define the neural network architecture
class Denoiser(nn.Module):

    @ignore_warnings
    def __init__(self):
        super(Denoiser, self).__init__()

        '''
        encoder
        '''

        self.e_fc1 = nn.Linear(28 * 28, 512)
        self.e_fc2 = nn.Linear(512, 256)
        self.e_fc3 = nn.Linear(256, 128)
        self.e_fc4 = nn.Linear(128, latent_space)

        '''
        decoder
        '''

        self.d_fc1 = nn.Linear(latent_space, 128)
        self.d_fc2 = nn.Linear(128, 256)
        self.d_fc3 = nn.Linear(256, 512)
        self.d_fc4 = nn.Linear(512, 28 * 28)
    
    @ignore_warnings
    def forward(self, x):
        
        '''
        encode
        '''
        x = F.relu(self.e_fc1(x))
        x = F.relu(self.e_fc2(x))
        x = F.relu(self.e_fc3(x))
        x = F.relu(self.e_fc4(x))

        z = x
        
        '''
        decode
        '''
        x = F.relu(self.d_fc1(x))
        x = F.relu(self.d_fc2(x))
        x = F.relu(self.d_fc3(x))
        x = torch.sigmoid(self.d_fc4(x))
                
        return x, z

# we have to add a custom loss function as a regularizing term
def loss_function(recon_x, x, z):
    # specify loss function
    criterion = nn.MSELoss()
    reconstuction_loss = criterion(recon_x, x.view(x.size(0), -1))

    regularizer = (torch.sum(z.pow(2))).pow(1/2)

    return reconstuction_loss + regularizer

@ignore_warnings
def run_denoiser():
    # initalize the neural network
    model = Denoiser()
    print(model)

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, n_epochs+1):
        # put the model in training mode
        model.train()

        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            
            # add random noise to the input images
            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)
                    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            '''
            forward pass: compute predicted outputs by passing *noisy* images to the model
            '''
            # reshape the input 
            noisy_imgs = noisy_imgs.view(noisy_imgs.size(0), -1)
            outputs, z = model(noisy_imgs)
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = loss_function(outputs, images.view(images.size(0), -1), z)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # put the model in evaluation mode
    model.eval()

    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # add noise to the test images
    noisy_imgs = images + noise_factor * torch.randn(*images.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)

    # get sample outputs
    output, z = model(noisy_imgs.view(noisy_imgs.size(0), -1))
    # prep images for display
    noisy_imgs = noisy_imgs.numpy()

    # output is resized into a batch of images
    output = output.view(batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    # input images on top row, reconstructions on bottom
    for noisy_imgs, row in zip([noisy_imgs, output], axes):
        for img, ax in zip(noisy_imgs, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

if __name__ == "__main__":
    run_denoiser()