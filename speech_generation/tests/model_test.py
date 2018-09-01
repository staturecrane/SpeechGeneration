import random

import numpy as np
import pytest
import torch

from speech_generation.speech_model.model import EncoderText
from speech_generation.utils import text_utils

TEST_DATA_DIR = 'speech_generation/tests/test_data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
HIDDEN_SIZE = 128
OUTPUT_SIZE = 100

@pytest.fixture
def text_encoder():
    text_encoder = EncoderText(MAX_LENGTH, 2).to(DEVICE)
    yield text_encoder


def test_text_encoder(text_encoder):
    sentence = 'this is a test sentence'
    input_vectors = text_utils.get_input_vectors(sentence, MAX_LENGTH).to(DEVICE)
    input_vectors = input_vectors.unsqueeze(0)
    output = text_encoder(input_vectors)
    print(output.size())
