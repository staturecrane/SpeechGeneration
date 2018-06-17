import random

import numpy as np
import torch
from torch.autograd import Variable

from speech_generation.speech_model.model  import EmbeddingRNN, AudioRNN
from speech_generation.speech_model.loader import LibriSpeech
from speech_generation.utils import text_utils

TEST_DATA_DIR = 'speech_generation/tests/test_data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 128
OUTPUT_SIZE = 100


def test_embedding_model():

    sample_dataset = text_utils.get_filenames_and_text(f'{TEST_DATA_DIR}/text1.txt')
    random_sample = random.choice(list(sample_dataset.items()))

    inputs = np.array([text_utils.convert_to_onehot(char) for char in random_sample[1]])
    input_tensors = torch.from_numpy(inputs).to(DEVICE)

    embedding_model = EmbeddingRNN(input_tensors.size(1), HIDDEN_SIZE, device=DEVICE).to(DEVICE)
    hidden = embedding_model.initHidden()

    output = embedding_model(Variable(input_tensors[0].unsqueeze(0)), hidden)
    assert output.size(1) == HIDDEN_SIZE


def test_audio_model():
    sample_input = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)

    audio_model = AudioRNN(HIDDEN_SIZE, OUTPUT_SIZE, device=DEVICE).to(DEVICE)
    hidden = audio_model.initHidden()

    output, stop, hidden = audio_model(Variable(sample_input), hidden)
    assert output.size(1) == OUTPUT_SIZE
    assert stop.size(1) == 1


def test_loader():
    loader = LibriSpeech('data')
    sample, (sample_rate, audiofile) = loader[0]

    assert isinstance(sample, str)
    assert sample_rate
    assert audiofile.numpy().any()

    for audio in audiofile.data:
        assert audio >= -1.0 or audio <= 1.0
