import math

import pytest
import tensorflow as tf
import torch

from speech_generation.config import ALL_LETTERS, N_LETTERS
from speech_generation.utils import utils, audio_utils, text_utils

TEST_DATA_DIR = 'speech_generation/tests/test_data'


def test_load_audio():
    sample_rate, audio = audio_utils.load_audio(f'{TEST_DATA_DIR}/test.wav')
    assert sample_rate == 16000
    audio_utils.save_audio('saved_output.wav', audio, sample_rate)


def test_load_text():
    sentence = 'test a sentence for me John'
    sen_length = len(sentence)
    max_length = 100

    vectors = text_utils.get_input_vectors(sentence, max_length=max_length)
    assert len(vectors) == max_length
    # confirm padded long elements equal N_LETTERS - 1
    assert all(filter(lambda x: x == N_LETTERS - 1, vectors[sen_length:]))
    reconstructed = [ALL_LETTERS[idx] for idx in vectors[:sen_length]]
    assert ''.join(reconstructed) == sentence
