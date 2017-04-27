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

transform = transforms.Compose([transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, transform=transform,download=True), batch_size=64,
                                           shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(100, 128)
        self.map3 = nn.Linear(128, 28 * 28)

    def forward(self, x):

        x = F.relu(self.map1(x))

        return F.sigmoid(self.map3(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(28 * 28 , 128)
        self.map3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = F.relu(self.map1(x))

        return self.map3(x)


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


d_steps = 5

D = Discriminator()
G = Generator()

#G_solver = optim.Adam(G.parameters(), lr=1e-3)
#D_solver = optim.Adam(D.parameters(), lr=1e-3)

G_solver = optim.RMSprop(G.parameters(), lr=5e-4)
D_solver = optim.RMSprop(D.parameters(), lr=5e-4)

ones_label = Variable(torch.ones(mb_size))
zeros_label = Variable(torch.zeros(mb_size))

bce = nn.BCELoss()

for e in range(20):

    avg_d_loss_real = 0
    avg_d_loss_fake = 0
    avg_g_loss = 0
    iterator = iter(train_loader)

    for i in range(len(train_loader) // d_steps):

        for d in range(d_steps):

            X, y = iterator.next()

            if X.size()[0] != 64:
                continue

            X, y = Variable(X) ,Variable(one_hot(y))
            X = X.view(-1, 28 * 28)

            z = Variable(torch.randn(mb_size, Z_dim))

            D.zero_grad()

            # Dicriminator forward-loss-backward-update
            G_sample = G(z).detach()
            D_real = D(X)
            D_fake = D(G_sample)

            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward()
            D_solver.step()

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        G.zero_grad()
        # Generator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        G_solver.step()

        #print(D_loss.data[0], G_loss.data[0])

    print('Iter-{}; D_loss: {}; G_loss: {}'.format(e, D_loss.data.numpy(), G_loss.data.numpy()))

    y = (torch.ones(64) * 7).long()
    y = Variable(one_hot(y))
    samples = G(z).data.numpy()[:16]

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
    c += 1
    plt.close(fig)

    torch.save(G, open('G.p', 'wb'))
    torch.save(D, open('D.p', 'wb'))

