import pytest
import tensorflow as tf
import torch

from speech_generation.config import ALL_LETTERS, N_LETTERS
from speech_generation.utils import utils, audio_utils, text_utils

TEST_DATA_DIR = 'speech_generation/tests/test_data'


def test_load_audio():
    sample_rate, audio = audio_utils.load_audio(f'{TEST_DATA_DIR}/test.wav')
    assert sample_rate == 16000
    assert audio.shape[0] > 0 and audio.shape[1] == 1


@pytest.fixture
def audio():
    return audio_utils.load_audio(f'{TEST_DATA_DIR}/test.wav')


def test_chunk_audio(audio):
    sample_rate, audio = audio
    chunksize = 100
    chunked_audio = audio_utils.chunk_audio(audio, chunksize=chunksize)
    if len(chunked_audio[-1]) < chunksize:
        assert (len(audio) - len(chunked_audio[-1])) % chunksize == 0
    else:
        assert len(audio) % chunksize == 0

    assert len(chunked_audio)


def test_save_audio(audio):
    sample_rate, audio = audio
    chunksize = 100
    chunked_audio = audio_utils.chunk_audio(audio, chunksize=chunksize)
    new_audio = []
    for x in chunked_audio:
        for sample in x:
            new_audio.append(sample)
    audio_utils.save_audio('test_out.wav', torch.Tensor(new_audio), sample_rate)


def test_create_dataset():
    text_files_string = ''.join(text_utils.get_text_files(TEST_DATA_DIR))
    assert 'text1.txt' in text_files_string and 'text2.txt' in text_files_string

    sample_dict = text_utils.get_filenames_and_text(f'{TEST_DATA_DIR}/text1.txt')
    for key, sample in sample_dict.items():
        assert len(key.split('-')) == 3
        assert sample.split(' ')
        assert sample.strip() == sample
        assert not '\n' in sample


def test_merge_dicts():
    sample_dict_one = text_utils.get_filenames_and_text(f'{TEST_DATA_DIR}/text1.txt')
    sample_dict_two = text_utils.get_filenames_and_text(f'{TEST_DATA_DIR}/text2.txt')

    merged = text_utils.merge_dicts(sample_dict_one, sample_dict_two)
    assert merged == {**sample_dict_one, **sample_dict_two}


def test_onehot():
    onehot = text_utils.convert_to_onehot('C')
    index = ALL_LETTERS.index('C')

    assert len(onehot) == N_LETTERS
    assert onehot[index] == 1

    assert not any(onehot[:index])
    assert not any(onehot[index + 1:])


def test_dataset():
    sample_sets = text_utils.load_dataset(TEST_DATA_DIR)
    assert isinstance(sample_sets, dict)
    for key, value in sample_sets.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
