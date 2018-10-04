from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from speech_generation.config import N_LETTERS


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        # self.embedding = nn.Embedding(input_size, 10)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # embedded = self.embedding(input)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


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
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(4, nf * 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 16, nf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 2, nf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class TextDecoder(nn.Module):
    def __init__(self, nf, z_dim, nc):
        super(TextDecoder, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_dim, nf * 32, 4, 1, 0, bias=False),
            # torch.nn.BatchNorm2d(nf * 64),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.ConvTranspose2d(nf * 64, nf * 32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nf * 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(nf * 32, nf * 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(nf * 16, nf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nf),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(nf, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nf, nc):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(1, nf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf, nf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf * 8, nf * 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf * 16, nf * 32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf * 32, nf * 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf * 64, 1, 4, 1, 0, bias=False, )
        )
        self.out_layer = nn.Linear(1173, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        return self.out_layer(output.view(output.size(0), -1))


class Encoder(nn.Module):
    def __init__(self, nf, nc):
        super(Encoder, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(1, nf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf, nf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(nf * 8, nf * 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(nf * 16, 4, 4, 2, 1, bias=False)
        )

    def forward(self, input):
        return self.main(input)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=294):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.tanh(self.out(output[0]))
        return output, hidden, attn_weights


class AttnDecoderAudioRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1, max_length=294):
        super(AttnDecoderAudioRNN, self).__init__()
        self.input_Size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.input_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.input_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        attn_weights = F.softmax(
            self.attn(torch.cat((input, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((input, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.tanh(self.out(output[0]))
        return output, hidden, attn_weights

# copypasta
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
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
