import math
import os
import random

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
    audio_binary, sample_rate = torchaudio.load(os.path.abspath(filepath), normalization=True)
    return sample_rate, audio_binary


def save_audio(outfile, sound, sample_rate):
    torchaudio.save(os.path.abspath(outfile), sound, sample_rate)


def reshape_audio(audio_tensor, hz, max_time=2, randomize=False):
    # max_length = int(hz * max_time)
    # max_length = 150528 -- 3 * 224 * 224 
    # max_length = 51200 -- 16000 * 3 ??
    max_length = 196608
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
        if randomize:
            max_idx = len(audio_tensor) - max_length - 1
            start_point = random.randint(0, max_idx)
            audio_tensor = audio_tensor[start_point: start_point + max_length]
        else:
            audio_tensor = audio_tensor[:max_length]
    return audio_tensor

