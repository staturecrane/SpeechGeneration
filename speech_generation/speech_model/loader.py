import json
import random

import torch
from torch.utils.data import Dataset
import torchaudio.transforms as transform

from speech_generation.utils import audio_utils, text_utils


def encode_audio(raw_audio):
    minval = raw_audio.min()
    maxval = raw_audio.max()
    audio = (((raw_audio - minval) / float(maxval - minval)) - 0.5) * 2.0
    return audio


def mu_encode(raw_audio, quantization_channels=255):
    mu = torch.Tensor([quantization_channels - 1])
    safe_abs = torch.abs(raw_audio)
    magnitude = torch.log1p(mu * safe_abs) / torch.log1p(mu)
    signal = torch.sign(raw_audio) * magnitude
    return ((signal + 1.0) / 2.0 * mu + 0.5)


def mu_decode(raw_audio, DEVICE, quantization_channels=255):
    mu = torch.Tensor([quantization_channels - 1]).to(DEVICE)
    signal = 2.0 * ((raw_audio) / mu) - 1.0
    magnitude = (1.0 / mu) * ((1.0 + mu)**abs(signal) - 1.0)
    return torch.sign(signal) * magnitude


class LibriSpeech(Dataset):
    def __init__(self, root_dir, device, max_time=5, max_length=400, randomize_speaker_samples=False):
        self.device = device
        self.max_length = max_length
        self.max_time = max_time
        self.randomize_speaker_samples = randomize_speaker_samples

        self.speaker_dict = json.loads(open('speaker_dict.json').read())

        dataset = text_utils.load_dataset(root_dir)
        self.dataset = [(key, value) for key, value in dataset.items()]
        self.root_dir = root_dir
        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filekey, sample = self.dataset[idx]

        speaker = filekey.split('-')[0]
        speaker_idx = self.speaker_dict[speaker]
        speaker_tensor = torch.zeros(2)
        speaker_tensor[speaker_idx] = 1.0
        sample_rate, audio = audio_utils.load_audio(f'{self.root_dir}/{filekey}.wav')
        char_inputs = text_utils.get_input_vectors(sample, max_length=self.max_length)
        audio_target = audio_utils.reshape_audio(audio,
                                                 sample_rate,
                                                 max_time=self.max_time,
                                                 randomize=self.randomize_speaker_samples)
        return audio_target.view(1, -1), speaker_tensor, char_inputs


class LibriSpeechVectors(Dataset):
    def __init__(self, root_dir, device, max_time=5, max_length=400, randomize_speaker_samples=False):
        self.device = device
        self.max_length = max_length
        self.max_time = max_time
        self.randomize_speaker_samples = randomize_speaker_samples

        self.speaker_dict = json.loads(open('speaker_dict.json').read())

        dataset = text_utils.load_dataset(root_dir)
        self.dataset = [(key, value) for key, value in dataset.items()]
        self.root_dir = root_dir
        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filekey, sample = self.dataset[idx]

        speaker = filekey.split('-')[0]
        speaker_idx = self.speaker_dict[speaker]
        speaker_tensor = torch.zeros(2)
        speaker_tensor[speaker_idx] = 1.0
        sample_rate, audio = audio_utils.load_audio(f'{self.root_dir}/{filekey}.wav')
        # char_inputs = text_utils.get_input_vectors(sample, max_length=self.max_length)
        char_inputs = torch.from_numpy(text_utils.get_input_word_vectors(sample, max_length=self.max_length))
        audio_target = audio_utils.reshape_audio(audio,
                                                 sample_rate,
                                                 max_time=self.max_time,
                                                 randomize=self.randomize_speaker_samples)
        return audio_target.view(1, -1), speaker_tensor, char_inputs