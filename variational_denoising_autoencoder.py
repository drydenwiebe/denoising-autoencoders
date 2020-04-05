from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import warnings
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np

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

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() and args.no_cuda else 'cpu')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 128
# if we use dropout or not
dropout = False
# define the learning rate
learning_rate = 1e-4
# number of epochs to train the model
n_epochs = 1500
# for adding noise to images
noise_factor = 0.5
# defines the size of the latent space
latent_space = 64
# weight decay for ADAM
weight_decay=1e-5
# set the seed for PyTorch
torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if not args.no_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc31 = nn.Linear(128, latent_space)
        self.fc32 = nn.Linear(128, latent_space)
        self.fc4 = nn.Linear(latent_space, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


model = VAE().to(device)
print(str(model) + '\n')

print("Running on: " + str(device) + '\n')

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, z):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    regularization = torch.sum(z.pow(2)).pow(0.5)

    return BCE + KLD + 100 * regularization

# Reconstruction loss only
def reconstruction_loss(recon_x, x, mu, logvar, z):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    return BCE

@ignore_warnings
def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        # add noise to the test images
        noisy_images = data + noise_factor * torch.randn(*data.shape)
        noisy_images = np.clip(noisy_images, 0., 1.)

        # transfer the training data to the correct device
        data = data.to(device)
        noisy_images = noisy_images.to(device)

        optimizer.zero_grad()

        outputs, mu, logvar, z = model.forward(noisy_images)

        loss = loss_function(outputs, data, mu, logvar, z)

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

@ignore_warnings
def test(epoch):
    model.eval()
    test_loss = 0
    test_loss_reconstruction = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            # add noise to the test images
            noisy_images = data + noise_factor * torch.randn(*data.shape)
            noisy_images = np.clip(noisy_images, 0., 1.)

            # transfer the training data to the correct device
            data = data.to(device)
            noisy_images = noisy_images.to(device)

            outputs, mu, logvar, z = model(noisy_images)

            test_loss += loss_function(outputs, data, mu, logvar, z).item()

            test_loss_reconstruction += reconstruction_loss(outputs, data, mu, logvar, z).item()

            if i == 0 and epoch % 100 == 0:
                for i in range(0, 2):
                    n = min(data.size(0), 8)
                    comparison = torch.cat([noisy_images[:n + n * i],
                                        outputs.view(batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                            'results/reconstruction_' + str(epoch) + ' ' + str(i) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    test_loss_reconstruction /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Test set reconstuction loss: {:.4f}'.format(test_loss_reconstruction))

if __name__ == "__main__":
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test(epoch)

        if epoch % 100 == 0:
            with torch.no_grad():
                for i in range(0, 2): 
                    sample = torch.randn(batch_size, latent_space).to(device)
                    sample = model.decode(sample).cpu()
                    save_image(sample.view(batch_size, 1, 28, 28),
                            'results/sample_' + str(epoch) + ' ' + str(i) + '.png')
