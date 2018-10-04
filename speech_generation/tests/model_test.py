import random

import numpy as np
import pytest
import torch

from speech_generation.speech_model.model import EncoderRNN, Decoder
from speech_generation.utils import text_utils

TEST_DATA_DIR = 'speech_generation/tests/test_data'
DEVICE = torch.device("cpu")

MAX_LENGTH = 100
HIDDEN_SIZE = 128
OUTPUT_SIZE = 100

@pytest.fixture
def encoder_rnn():
    encoder_rnn_instance = EncoderRNN(128, 256, DEVICE).to(DEVICE)
    yield encoder_rnn_instance


def test_model_outputs(encoder_rnn):
    test_input = torch.Tensor(1, 1, 128).to(DEVICE).long()
    hidden = torch.Tensor(1, 1, 256).to(DEVICE)

    for i in range(test_input.shape[0]):
        output, hidden = encoder_rnn(test_input[i], hidden)
        assert output.size() == (1, 128, 256)
        assert hidden.size() == (1, 1, 256)
