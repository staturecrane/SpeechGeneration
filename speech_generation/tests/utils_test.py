import pytest
import tensorflow

from speech_generation.utils import load_audio, chunk_audio, get_text_files, get_filenames_and_text


TEST_DATA_DIR = 'speech_generation/tests/test_data'


def test_load_audio():
    sample_rate, audio = load_audio(f'{TEST_DATA_DIR}/test.wav')
    assert sample_rate == 16000
    assert audio.shape[0] > 0 and audio.shape[1] == 1


@pytest.fixture
def audio():
    return load_audio(f'{TEST_DATA_DIR}/test.wav')


def test_chunk_audio(audio):
    sample_rate, audio = audio
    chunksize = 100
    chunked_audio = chunk_audio(audio, chunksize=chunksize)
    if len(chunked_audio[-1]) < chunksize:
        assert (len(audio) - len(chunked_audio[-1])) % chunksize == 0
    else:
        assert len(audio) % chunksize == 0

    assert len(chunked_audio)


def test_create_dataset():
    text_files_string = ''.join(get_text_files(TEST_DATA_DIR))
    assert 'text1.txt' in text_files_string and 'text2.txt' in text_files_string

    sample_dict = get_filenames_and_text(f'{TEST_DATA_DIR}/text1.txt')
    for key, sample in sample_dict.items():
        assert len(key.split('-')) == 3
        assert len(sample.split(' ')) > 0
        assert sample.strip() == sample
        assert not '\n' in sample
