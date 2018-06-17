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


class AudioRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(AudioRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)
        self.lin_stop = nn.Linear(hidden_size, 1)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_tensor, hidden):
        hidden = self.gru(input_tensor, hidden)
        output = self.lin(hidden)
        stop = self.lin_stop(hidden)

        return self.tanh(output), self.sig(stop), hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=self.device)
