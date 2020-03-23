import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import warnings
from functools import wraps

'''
Based on the template from: https://github.com/pytorch/examples/tree/master/vae

Refrence: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/denoising-autoencoder/Denoising_Autoencoder_Solution.ipynb
Refrence: https://github.com/milindmalshe/Fully-Connected-Neural-Network-MNIST-Classification-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
Refrence: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

'''

# ignoring Pyorch warnings
def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 128
# if we use dropout or not
dropout = False
# define the learning rate
learning_rate = 0.000001
# number of epochs to train the model
n_epochs = 400
# for adding noise to images
noise_factor = 0.5
# defines the size of the latent space
latent_space_size = 40
# determines the loss function used for the reconstruction loss
loss_func = 'BCE'
# weight decay for ADAM
weight_decay = 1e-5
# interval for printing
log_interval = 100

# load the training and test datasets
train_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# define the neural network architecture
class ConvolutionalDenoiser(nn.Module):
    def __init__(self):
        super(ConvolutionalDenoiser, self).__init__()

        '''
        encoder
        '''
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # create a [batch_size, latent_space_size vector]
        self.conv_latent = nn.Conv2d(8, 1, 3, padding=1)

        self.fc1_1 = nn.Linear(49, latent_space_size)
        self.fc1_2 = nn.Linear(49, latent_space_size)

        '''
        decoder
        '''

        # layer from the decoder that takes the latent space 
        self.fc2 = nn.Linear(latent_space_size, 49)
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(1, 8, 3, padding=1)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def encode(self, x):
        '''
        encode
        '''
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        #x = self.pool(x)  # compressed representation

        x = F.relu(self.conv_latent(x).view(-1, 49))

        return self.fc1_1(x), self.fc1_2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):

        x = F.relu(self.fc2(z))
        x = x.view(x.shape[0], 1, 7, 7)

        '''
        decode
        '''
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = torch.sigmoid(self.conv_out(x))

        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
@ignore_warnings
def loss_function(recon_x, x, mu, logvar):

    if(loss_func == 'BCE'):
        reconstruction_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    elif(loss_func == 'MSE'):
        loss = nn.MSELoss()
        reconstruction_loss = loss(recon_x.view(-1, 784), x.view(-1, 784))
    elif(loss_func == 'L1'):
        loss = nn.L1Loss(reduction='sum')
        reconstruction_loss = loss(recon_x.view(-1, 784), x.view(-1, 784))
    else:
        reconstruction_loss = None
        raise ValueError('Invalid loss function')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + KLD

def run_denoiser():
    # initalize the neural network
    model = ConvolutionalDenoiser()
    print(model)

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        train_image_count = 0

        ###################
        # train the model #
        ###################
        for batch_idx, data in enumerate(train_loader):
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            
            ## add random noise to the input images
            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)
                    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            ## forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs, mu, logvar = model.forward(noisy_imgs)
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = loss_function(outputs, images, mu, logvar)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()
            train_image_count += images.shape[0]

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    batch_idx * batch_size / len(train_loader.dataset),
                    loss.item() / batch_size))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch,
            train_loss / train_image_count))
    
    model.eval()
    
    test_loss = 0.0
    test_image_count = 0

    for test_batch, data in enumerate(test_loader):
        images, _ = data
        
        ## add random noise to the input images
        noisy_imgs = images + noise_factor * torch.randn(*images.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        ## forward pass: compute predicted outputs by passing *noisy* images to the model
        outputs, mu, logvar = model.forward(noisy_imgs)
        # calculate the loss
        # the "target" is still the original, not-noisy images
        loss = loss_function(outputs, images, mu, logvar)

        test_loss += loss.item()
        test_image_count += images.shape[0]

        print('Test Batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            test_batch,
            test_batch * batch_size,
            len(test_loader.dataset),
            test_batch * batch_size / len(test_loader.dataset),
            loss.item() / batch_size))

    print('Test loss: {:.4f}'.format(
        test_loss / test_image_count))

    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # add noise to the test images
    noisy_imgs = images + noise_factor * torch.randn(*images.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)

    # get sample outputs
    output, _, _ = model.forward(noisy_imgs)
    # prep images for display
    noisy_imgs = noisy_imgs.numpy()

    # output is resized into a batch of iages
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
    plt.clf()

    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # add noise to the test images
    noisy_imgs = images + noise_factor * torch.randn(*images.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)

    # get sample outputs
    output, _, _ = model.forward(noisy_imgs)
    # prep images for display
    noisy_imgs = noisy_imgs.numpy()

    # output is resized into a batch of iages
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

    # here we make the generated samples

if __name__ == "__main__":
    run_denoiser()
