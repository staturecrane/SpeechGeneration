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
from speech_generation.speech_model.model import Encoder, Decoder
from speech_generation.speech_model.loader import LibriSpeech

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACTOR = 2**31
SOS_token = config.N_LETTERS - 1


def main(cfg_path):
    cfg_file = open(os.path.abspath(cfg_path))
    cfg = yaml.load(cfg_file.read())
    data_dir = cfg.get('data')
    test_data_dir = cfg.get('test_data')

    try:
        assert data_dir
    except AssertionError:
        raise 'No "dataset_dir" found in config'

    hidden_size = cfg.get('hidden_size', 128)
    batch_size = cfg.get('batch_size', 10)
    max_length = cfg.get('max_length', 400)
    max_time = cfg.get('max_time', 5)

    dataset = LibriSpeech(data_dir, DEVICE, scale_factor=FACTOR, max_time=max_time, max_length=max_length)
    test_dataset = LibriSpeech(test_data_dir, DEVICE, scale_factor=FACTOR, max_time=max_time, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    encoder = Encoder(64, 1).to(DEVICE)
    decoder = Decoder(hidden_size, 64, 1).to(DEVICE)

    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    mse = torch.nn.MSELoss()

    for epoch in range(100):
        for sample_idx, (audio, gender, text) in enumerate(dataloader):
            audio, gender, text = audio.to(DEVICE), gender.to(DEVICE), text.to(DEVICE)

            encoder.zero_grad()
            decoder.zero_grad()

            optimizer_e.zero_grad()
            optimizer_d.zero_grad()

            batch_size = audio.size(0)

            _, hidden = encoder(audio)
            audio_output = decoder(hidden)

            err = mse(audio_output, audio)
            err.backward()

            optimizer_e.step()
            optimizer_d.step()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx} -- err: {err.item()}")
                sample(
                    encoder,
                    decoder,
                    'generated',
                    epoch,
                    sample_idx,
                    16000,
                    *random.choice(test_dataset)
                )

            if sample_idx % cfg.get('save_iter', 1000) == 0:
                with open('encoder_conv.pt', 'wb') as enc_file:
                    torch.save(encoder.state_dict(), enc_file)

                with open('decoder_conv.pt', 'wb') as dec_file:
                    torch.save(decoder.state_dict(), dec_file)


def sample(encoder, decoder, outfolder, epoch, sample_idx, sample_rate, audio, gender, text):
    with torch.no_grad():
        audio = audio.view(1, 1, -1)
        audio = audio.to('cuda' if torch.cuda.is_available() else 'cpu')
        _, hidden = encoder(audio)
        audio_output = decoder(hidden)

    torchaudio.save("{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx), audio_output.view(-1).cpu(), sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
