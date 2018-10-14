# 4 x 2352
import torch
import torch.nn as nn

nf = 2

class TextDecoder(nn.Module):
    def __init__(self, nf, z_dim, nc):
        super(TextDecoder, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(z_dim, nf * 64, 4, 2, 1, dilation=2, bias=False),
            torch.nn.BatchNorm1d(nf * 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 64, nf * 32, 4, 2, 1, dilation=4, bias=False),
            torch.nn.BatchNorm1d(nf * 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 32, nf * 16, 4, 2, 1, dilation=8, bias=False),
            torch.nn.BatchNorm1d(nf * 16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 16, nf * 8, 4, 2, 1, dilation=16, bias=False),
            torch.nn.BatchNorm1d(nf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 8, nf * 4, 4, 2, 1, dilation=32, bias=False),
            torch.nn.BatchNorm1d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 4, nf * 2, 4, 2, 1, dilation=64, bias=False),
            torch.nn.BatchNorm1d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf * 2, nf, 4, 2, 1, dilation=128, bias=False),
            torch.nn.BatchNorm1d(nf),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose1d(nf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    def __init__(self, nf, nc):
        super(Encoder, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(1, nf, 4, 2, 1, bias=False, dilation=2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf, nf * 2, 4, 2, 1, bias=False, dilation=4),
            torch.nn.BatchNorm1d(nf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(nf * 2, nf * 4, 4, 2, 1, bias=False, dilation=8),
            torch.nn.BatchNorm1d(nf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout(0.2),
            # torch.nn.Conv1d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm1d(nf * 8),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout(0.2),
            # torch.nn.Conv1d(nf * 8, nf * 16, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm1d(nf * 16),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout(0.2),
            torch.nn.Conv1d(nf * 4, 1, 4, 2, 1, bias=False)
        )

    def forward(self, input):
        return self.main(input)

embedding = nn.Embedding(2, 10)

male = torch.LongTensor([1])

embedded = embedding(male).view(1, 1, -1)

encoder = Encoder(nf, 1)
test = torch.zeros(1, 1, (100 * 50) + 10)
output = encoder(torch.cat((test, embedded), 2))
print(output.size())

decoder = TextDecoder(2, 1, 1)

output = decoder(output)
print(output.size())
