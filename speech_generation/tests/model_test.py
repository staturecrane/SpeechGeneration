import random

import numpy as np
import torch
from torch.autograd import Variable

from speech_generation.speech_model  import EmbeddingRNN, AudioRNN
import speech_generation.utils as utils

TEST_DATA_DIR = 'speech_generation/tests/test_data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 128
OUTPUT_SIZE = 100


def test_embedding_model():

    sample_dataset = utils.get_filenames_and_text(f'{TEST_DATA_DIR}/text1.txt')
    random_sample = random.choice(list(sample_dataset.items()))

    inputs = np.array([utils.convert_to_onehot(char) for char in random_sample[1]])
    input_tensors = torch.from_numpy(inputs)

    embedding_model = EmbeddingRNN(input_tensors.size(1), HIDDEN_SIZE, device=DEVICE)
    hidden = embedding_model.initHidden()

    output = embedding_model(Variable(input_tensors[0].unsqueeze(0)), hidden)
    assert output.size(1) == HIDDEN_SIZE


def test_audio_model():
    sample_input = torch.zeros(1, HIDDEN_SIZE)

    audio_model = AudioRNN(HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, device=DEVICE)
    hidden = audio_model.initHidden()

    output = audio_model(Variable(sample_input), hidden)
    assert output.size(1) == OUTPUT_SIZE
