import torch
import torch.nn as nn
import torch.nn.functional as F

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
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=400):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AudioCNN(nn.Module):
    def __init__(self, nz, nf, nc):
        super(AudioCNN, self).__init__()
        self.in_layer = nn.Linear(nz, 610)
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(1, nf*64, 4, 1, 0, bias=False),
            torch.nn.BatchNorm1d(nf * 64),
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
        i2h = i2h.view(input.size(0), 1, -1)
        return self.main(i2h)


class Decoder(nn.Module):
    def __init__(self, nz, nf, nc):
        super(Decoder, self).__init__()
        self.main = torch.nn.Sequential(
            # torch.nn.ConvTranspose1d(1, nf*64, 4, 1, 0, bias=False),
            # torch.nn.BatchNorm1d(nf * 64),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.ConvTranspose1d(1, nf*32, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm1d(nf * 32),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(1, nf*16, 4, 2, 1, bias=False),
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
        return self.main(input)


class Classifier(nn.Module):
    def __init__(self, nf, nc, num_classes):
        super(Classifier, self).__init__()
        self.main = torch.nn.Sequential(
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
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*8, nf*16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*16, nf*32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*32, nf*64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*64, 1, 4, 1, 0, bias=False, )
        )
        self.out_layer = nn.Linear(1235, num_classes)
        self.soft = nn.Softmax()

    def forward(self, input):
        output = self.main(input)
        return self.out_layer(output.view(output.size(0), -1))


class Discriminator(nn.Module):
    def __init__(self, nf, nc):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
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
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*8, nf*16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*16, nf*32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*32, nf*64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*64, 1, 4, 1, 0, bias=False, )
        )
        self.out_layer = nn.Linear(622, 1)
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
            torch.nn.Conv1d(nf, nf*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf*2, nf*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(nf*4, nf*8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(nf*8, nf*16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(nf*16, 1, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm1d(nf * 32),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout(0.2),
            # torch.nn.Conv1d(nf*32, nf*64, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm1d(nf * 64),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Conv1d(nf*64, 1, 4, 1, 0, bias=False, )
        )

    def forward(self, input):
        return self.main(input)

