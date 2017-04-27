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
import torch.nn.parallel


class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)


class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid.{0}.relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


imageSize = 64

transform=transforms.Compose([transforms.ToTensor()])

transform=transforms.Compose([
                                   transforms.Scale(imageSize),
                                   transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])

cifar = datasets.CIFAR10('../cifar', train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(cifar, batch_size=64, shuffle=True)


def one_hot(target):
    y = torch.zeros(target.size()[0], 10)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y

cuda = False
mb_size = 64
Z_dim = 100
h_dim = 128
c = 0
lr = 1e-3
d_steps =5
nz= 100
nc = 3
ndf = 64
ngf = 64
ngpu = 0
n_extra_layers = 0

D = DCGAN_D(isize = imageSize, nz = nz, nc = nc, ndf = ndf, ngpu = ngpu, n_extra_layers=n_extra_layers)
G = DCGAN_G(isize = imageSize, nz = nz, nc = nc, ngf = ngf, ngpu = ngpu, n_extra_layers=n_extra_layers)

if cuda:
    D = D.cuda()
    G = G.cuda()

print(D)

G_solver = optim.RMSprop(G.parameters(), lr=5e-5)
D_solver = optim.RMSprop(D.parameters(), lr=5e-5)

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

            if X.size()[0] != mb_size:
                continue

            X = Variable(X)
            y = Variable(one_hot(y))
            z = Variable(torch.randn(mb_size, nz, 1, 1))

            if cuda:
                X= X.cuda()
                y = y.cuda()
                z = z.cuda()

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
        z = Variable(torch.randn(mb_size, nz, 1, 1))

        if cuda:
            z = z.cuda()

        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        G_solver.step()

    print('Iter-{}; D_loss: {}; G_loss: {}'.format(i, D_loss.data.numpy(), G_loss.data.numpy()))

    y = (torch.ones(64) * 7).long()
    y = Variable(one_hot(y))
    samples = G(z)[:30].data.mul(0.5).add(0.5)

    torchvision.utils.save_image(samples.data, 'out/{}.png'.format(str(c).zfill(3)), nrow=5, padding=2)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    c += 1

    torch.save(G, open('G.p', 'wb'))
    torch.save(D, open('D.p', 'wb'))
