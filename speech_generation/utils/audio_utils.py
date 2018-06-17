import torchaudio
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import os


def load_audio(filepath):
    """
    Loads and returns sample rate and audio for a specific file

    Args:
        filepath (string): Path to .wav audio file to open

    Returns:
        (int, list): sample rate and list of audio samples
    """
    audio_binary, sample_rate = torchaudio.load(os.path.abspath(filepath))
    return sample_rate, audio_binary


def save_audio(outfile, sound, sample_rate):
    torchaudio.save(os.path.abspath(outfile), sound, sample_rate)


def chunk_audio(audio, chunksize=100):
    """
    Chunks out list of audio samples into separate 1x100 lists

    Args:
        audio (list): list of audio samples from a single .wav file
        chunksize (int): optional size of chunks

    Returns:
        (int, list): sample rate and list of audio samples
    """
    return [audio[i:i+chunksize] for i in range(0, len(audio), chunksize)]
