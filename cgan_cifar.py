import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import os
import matplotlib.gridspec as gridspec


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.latent_vars = 100

        self.g_fc = nn.Linear(self.latent_vars, 512)

        self.dconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.dconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.dconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.dconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, output_padding=1)

        self.bn4 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.relu(self.g_fc(x))
        x = x.view(-1, 512, 1, 1)
        x = F.relu(self.bn2(self.dconv1(x)))
        x = F.relu(self.bn3(self.dconv2(x)))
        x = F.relu(self.bn4(self.dconv3(x)))
        x = F.sigmoid(self.dconv4(x))

        return (x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.d_fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.conv1(x), .2))
        x = self.bn2(F.leaky_relu(self.conv2(x), .2))
        x = self.bn3(F.leaky_relu(self.conv3(x), .2))
        x = self.bn4(F.leaky_relu(self.conv4(x), .2))
        x = x.view(-1, 512)
        x = F.sigmoid(self.d_fc(x))

        return (x)


def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

transform=transforms.Compose([transforms.ToTensor()])
cifar = datasets.CIFAR10('../cifar', train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(cifar, batch_size=64, shuffle=True)



def one_hot(target):
    y = torch.zeros(target.size()[0], 10)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y


mb_size = 64
Z_dim = 100
h_dim = 128
c = 0
lr = 1e-3

D = Discriminator()
G = Generator()

G_solver = optim.Adam(G.parameters(), lr=1e-3)
D_solver = optim.Adam(D.parameters(), lr=1e-3)

ones_label = Variable(torch.ones(mb_size))
zeros_label = Variable(torch.zeros(mb_size))

bce = nn.BCELoss()

for e in range(1):

    avg_d_loss_real = 0
    avg_d_loss_fake = 0
    avg_g_loss = 0
    iterator = iter(train_loader)

    #for i in range(len(train_loader)):
    for i in range(10):

        X, y = iterator.next()

        if X.size()[0] != mb_size:
            continue

        X = Variable(X)

        y = Variable(one_hot(y))

        z = Variable(torch.randn(mb_size, Z_dim))

        D.zero_grad()

        # Dicriminator forward-loss-backward-update
        G_sample = G(z).detach()
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss_real = F.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_loss.backward()
        D_solver.step()

        # Housekeeping - reset gradient

        G.zero_grad()
        # Generator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = F.binary_cross_entropy(D_fake, ones_label)

        G_loss.backward()
        G_solver.step()

        print(D_loss.data.numpy(), G_loss.data.numpy())

    print('Iter-{}; D_loss: {}; G_loss: {}'.format(i, D_loss.data.numpy(), G_loss.data.numpy()))

    y = (torch.ones(64) * 7).long()
    y = Variable(one_hot(y))
    samples = G(z)[:30]


    img = torchvision.utils.make_grid( samples.data)

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1,2,0)))
    #plt.show()

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
    c += 1
    plt.close()

    torch.save(G, open('G.p', 'wb'))
    torch.save(D, open('D.p', 'wb'))
