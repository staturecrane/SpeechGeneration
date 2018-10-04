# 4 x 2352
import torch
import torch.nn as nn

nf = 2

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
            torch.nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

decoder = TextDecoder(2, 200, 3)

test = torch.zeros(1, 200, 1, 1)
output = decoder(test)
print(output.size())