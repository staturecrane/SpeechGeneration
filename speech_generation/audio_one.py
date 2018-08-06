import os
import random
import sys
sys.path.append('.')

import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm
import yaml

from speech_generation import config
from speech_generation.utils import text_utils, audio_utils
from speech_generation.speech_model.model import Classifier
from speech_generation.speech_model.loader import LibriSpeech

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACTOR = 2**31
SOS_token = config.N_LETTERS - 1

def main(cfg_path):
    cfg_file = open(os.path.abspath(cfg_path))
    cfg = yaml.load(cfg_file.read())
    data_dir = cfg.get('dataset_dir')

    try:
        assert data_dir
    except AssertionError:
        raise 'No "dataset_dir" found in config'

    batch_size = cfg.get('batch_size', 10)
    max_length = cfg.get('max_length', 400)
    max_time = cfg.get('max_time', 5)
    learning_rate = cfg.get('learning_rate', 1e-3)

    dataset = LibriSpeech(data_dir, DEVICE, scale_factor=FACTOR, max_time=max_time, max_length=max_length, randomize_speaker_samples=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    classifier = Classifier(128, 1).to(DEVICE)
    optimizer_d = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        for sample_idx, (audio, gender, text) in enumerate(dataloader):
            audio, gender, text = audio.to(DEVICE), gender.to(DEVICE), text.to(DEVICE)

            classifier.train()
            classifier.zero_grad()

            _, d_output_real = classifier(audio)
            err_d = criterion(d_output_real, gender)
            err_d.backward()

            optimizer_d.step()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx}: errD: {err_d.item()}")

            if sample_idx % cfg.get('save_iter', 1000) == 0:
                with open('classifier.pt', 'wb') as audio_file:
                    torch.save(classifier.state_dict(), audio_file)


if __name__ == '__main__':
    main(sys.argv[1])
