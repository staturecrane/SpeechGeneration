from collections import namedtuple

import torch
import torch.nn as nn

from torchvision import models

from speech_generation.config import N_LETTERS


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

# copypasta
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad1d(1),
                      nn.Conv1d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad1d(1),
                      nn.Conv1d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Decoder(nn.Module):
    def __init__(self, nf, nc):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(4, nf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(nf * 16, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(nf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf * 8, nf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf * 16, nf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf * 32, nf * 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf * 64, 1, 4, 1, 0, bias=False, )
        )
        self.out_layer = nn.Linear(1173, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        return self.out_layer(output.view(output.size(0), -1))


class Encoder(nn.Module):
    def __init__(self, nf, nc):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(nf * 8, nf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(nf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(nf * 16, 4, 4, 2, 1, bias=False)
        )

    def forward(self, input):
        return self.main(input)

# copypasta
class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
