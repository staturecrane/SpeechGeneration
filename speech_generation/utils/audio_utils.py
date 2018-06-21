import math
import os

import torch
import torchaudio
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio


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


def reshape_audio(audio_tensor, hz, max_time=20):
    max_length = int(hz * max_time)
    audio_tensor = audio_tensor.view(-1)
    if len(audio_tensor) < max_length:
        new_audio = torch.zeros(max_length)
        for i in range(max_length):
            try:
                new_audio[i] = audio_tensor[i]
            except IndexError:
                new_audio[i] = 0
        audio_tensor = new_audio
    elif len(audio_tensor) > max_length:
        audio_tensor = audio_tensor[:max_length]
    return audio_tensor
