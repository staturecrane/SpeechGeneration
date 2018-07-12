import json
import random

import torch
from torch.utils.data import Dataset
import torchaudio.transforms as transform

from speech_generation.utils import audio_utils, text_utils


class LibriSpeech(Dataset):
    def __init__(self, root_dir, device, max_time=5, max_length=400, scale_factor=2**31, randomize_speaker_samples=False):
        self.device = device
        self.max_length = max_length
        self.max_time = max_time
        self.randomize_speaker_samples = randomize_speaker_samples

        self.speaker_dict = json.loads(open('speaker_dict.json').read())

        dataset = text_utils.load_dataset(root_dir)
        self.dataset = [(key, value) for key, value in dataset.items()]
        self.root_dir = root_dir
        self.scale = transform.Compose([transform.Scale(factor=scale_factor)])
        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filekey, sample = self.dataset[idx]
        speaker = filekey.split('-')[0]
        speaker_idx = self.speaker_dict[speaker]
        sample_rate, audio = audio_utils.load_audio(f'{self.root_dir}/{filekey}.wav')
        audio = self.scale(audio)

        char_inputs = text_utils.get_input_vectors(sample, max_length=self.max_length)
        audio_target = audio_utils.reshape_audio(audio, sample_rate, max_time=self.max_time, randomize=self.randomize_speaker_samples).view(1, -1)
        return audio_target, speaker_idx, char_inputs

