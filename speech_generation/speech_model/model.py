import torch
import torch.nn as nn

from speech_generation.config import N_LETTERS, MAX_INPUT_LENGTH


class EmbeddingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EmbeddingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.gru = nn.GRUCell(input_size, hidden_size)

    def forward(self, input_tensor, hidden):
        output = self.gru(input_tensor, hidden)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=self.device)


class AudioCNN(nn.Module):
    def __init__(self, nz, nf, nc):
        super(AudioCNN, self).__init__()
        self.in_layer = nn.Linear(nz, 247)
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(1, nf*64, 4, 1, 0, bias=False),
            torch.nn.BatchNorm1d(nf * 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf*64, nf*32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf*32, nf*16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf*16, nf*8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf*8, nf*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf*4, nf*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf*2, nf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf, 1, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        i2h = self.in_layer(input)
        return self.main(i2h.unsqueeze(0))


class Discriminator(nn.Module):
    def __init__(self, nf, nc):
        super(Discriminator, self).__init__()
        self.one = torch.nn.Sequential(
            torch.nn.Conv1d(1, nf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf, nf*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*2, nf*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*4, nf*8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )
        self.two = torch.nn.Sequential(
            torch.nn.Conv1d(nf*8, nf*16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*16, nf*32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*32, nf*64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*64, 1, 4, 1, 0)
        )
        self.out_layer = nn.Linear(497, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        one = self.one(input)
        output = self.two(one)
        return one, self.out_layer(output.view(1, -1))
