import pytest
import tensorflow

from speech_generation.utils import load_audio, build_model, chunk_audio


def test_load_audio():
    sample_rate, audio = load_audio('speech_generation/tests/test.wav')
    assert sample_rate == 16000
    assert audio.shape[0] > 0 and audio.shape[1] == 1


@pytest.fixture
def audio():
    return load_audio('speech_generation/tests/test.wav')


def test_chunk_audio(audio):
    sample_rate, audio = audio
    chunksize = 100
    chunked_audio = chunk_audio(audio, chunksize=chunksize)
    if len(chunked_audio[-1]) < chunksize:
        assert (len(audio) - len(chunked_audio[-1])) % chunksize == 0
    else:
        assert len(audio) % chunksize == 0

    assert len(chunked_audio)
def test_build_model():
    model = build_model()
    assert model


