import random

from torch.utils.data import Dataset
import torchaudio.transforms as transform

from speech_generation.utils import audio_utils, text_utils


class LibriSpeech(Dataset):
    def __init__(self, root_dir, scale_factor=2**31):
        dataset = text_utils.load_dataset(root_dir)
        self.dataset = [(key, value) for key, value in dataset.items()]
        self.root_dir = root_dir
        self.scale = transform.Compose([transform.Scale(factor=scale_factor)])
        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filekey, sample = self.dataset[idx]
        sample_rate, audio = audio_utils.load_audio(f'{self.root_dir}/{filekey}.wav')
        return sample, (sample_rate, self.scale(audio))
