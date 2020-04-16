from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

import warnings
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from PIL import Image as PILImage

'''
Colour MNIST based on the template from: https://github.com/wouterbulten/deeplearning-resources/tree/master/notebooks

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

# setup
# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() and args.no_cuda else 'cpu')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 512
# if we use dropout or not
dropout = False
# define the learning rate
learning_rate = 5e-4
# number of epochs to train the model
n_epochs = 250
# for adding noise to images
noise_factor = 0.5
# defines the size of the latent space
latent_space = 512
# weight decay for ADAM
weight_decay=1e-6
# set the seed for PyTorch
torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
    
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

# Read Lena image
lena = PILImage.open('resources/calibration.jpg')

def transfor_mnist_batch(batch_raw, batch_size=128, change_colors=True, show=False):
    batch_raw = np.transpose(batch_raw, (0, 2, 3, 1))

    batch_size = batch_raw.shape[0]

    # Resize
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in batch_raw])

    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)

    # Make binary
    batch_binary = (batch_rgb > 0.5)
    
    batch = np.zeros((batch_size, 64, 64, 3))

    for i in range(0, batch_size):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        image = np.asarray(image) / 255.0

        if change_colors:
            # Change color distribution
            for j in range(3):
                image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]
        
        batch[i] = image
    
    if show:
        count = 20
        plt.figure(figsize=(15,3))
        for i in range(count):
            plt.subplot(2, count // 2, i+1)
            plt.imshow(batch[i])
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    # return the batch into a shape that is useable by the network
    batch = np.transpose(batch, (0, 3, 1, 2))

    return torch.tensor(batch).float()

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 16 * 16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 512)
        self.fc22 = nn.Linear(512, 512)

        # Decoder
        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 16 * 16 * 16)
        self.fc_bn4 = nn.BatchNorm1d(16 * 16 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 16 * 16 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))

        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 16, 16)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))

        return torch.sigmoid(self.conv8(conv7).view(-1, 3, 64, 64))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def generate(self):
        z = noise = torch.randn(512, dtype=torch.float)
        return self.decode(z)

# define the loss functions
# reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, z):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 64 * 64 * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    regularization = torch.sum(torch.abs(z))

    return BCE + KLD + 10 * regularization

# reconstuction loss only
def reconstruction_loss(recon_x, x, mu, logvar, z):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 64 * 64 * 3), reduction='sum')

    return BCE

# define the training loop
@ignore_warnings
def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        # change the data to colour data
        data = transfor_mnist_batch(data)

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
            # change the data to colour data
            data = transfor_mnist_batch(data)

            # add noise to the test images
            noisy_images = data + noise_factor * torch.randn(*data.shape)
            noisy_images = np.clip(noisy_images, 0., 1.)

            # transfer the training data to the correct device
            data = data.to(device)
            noisy_images = noisy_images.to(device)

            outputs, mu, logvar, z = model(noisy_images)

            test_loss += loss_function(outputs, data, mu, logvar, z).item()

            test_loss_reconstruction += reconstruction_loss(outputs, data, mu, logvar, z).item()

            if i == 0 and epoch % 50 == 0:
                for i in range(0, 2):
                    n = min(data.size(0), 8)
                    comparison = torch.cat([noisy_images[:n + n * i],
                                        outputs.view(batch_size, 3, 64, 64)[:n]])
                    save_image(comparison.cpu(),
                            'results/reconstruction_' + str(epoch) + ' ' + str(i) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    test_loss_reconstruction /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Test set reconstuction loss: {:.4f}'.format(test_loss_reconstruction))

model = VAE().to(device)
print(str(model) + '\n')

print("Running on: " + str(device) + '\n')

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if __name__ == "__main__":
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test(epoch)

        if epoch % 50 == 0:
            with torch.no_grad():
                for i in range(0, 2): 
                    sample = torch.randn(128, latent_space).to(device)
                    sample = model.decode(sample).cpu()
                    save_image(sample.view(128, 3, 64, 64),
                            'results/sample_' + str(epoch) + ' ' + str(i) + '.png')
