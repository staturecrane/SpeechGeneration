import os
import random
import sys
sys.path.append('.')

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm
import yaml

from speech_generation import config
from speech_generation.utils import text_utils, audio_utils
from speech_generation.speech_model.model import EncoderRNN, Decoder
from speech_generation.speech_model.loader import LibriSpeech

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACTOR = 32768
SOS_token = config.N_LETTERS - 1


def main(cfg_path):
    cfg_file = open(os.path.abspath(cfg_path))
    cfg = yaml.load(cfg_file.read())
    data_dir = cfg.get('data')
    test_data_dir = cfg.get('test-data')

    try:
        assert data_dir
    except AssertionError:
        raise 'No "dataset_dir" found in config'

    hidden_size = cfg.get('hidden_size', 128)
    batch_size = cfg.get('batch_size', 10)
    max_length = cfg.get('max_length', 400)
    max_time = cfg.get('max_time', 5)
    num_classes = cfg.get('num_classes')

    if not num_classes:
        raise AssertionError('Number of classes not provided')

    learning_rate = cfg.get('learning_rate', 1e-3)

    dataset = LibriSpeech(data_dir, DEVICE, scale_factor=FACTOR, max_time=max_time, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = LibriSpeech(data_dir, DEVICE, scale_factor=FACTOR, max_time=max_time, max_length=max_length)

    encoder = EncoderRNN(config.N_LETTERS, hidden_size, DEVICE).to(DEVICE)
    encoded_state = torch.load('encoder.pt')
    encoder.load_state_dict(encoded_state)

    decoder = Decoder(hidden_size, 4, 1).to(DEVICE)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer_g = torch.optim.Adam(parameters, lr=1e-3)

    bce = torch.nn.BCELoss()
    mse = torch.nn.MSELoss()

    for epoch in range(100):
        for sample_idx, (audio, speaker, text) in enumerate(dataloader):
            audio, speaker, text = audio.to(DEVICE), speaker.to(DEVICE), text.to(DEVICE)

            encoder.zero_grad()
            decoder.zero_grad()
            optimizer_g.zero_grad()

            batch_size = audio.size(0)

            encoder_hidden = encoder.initHidden()

            for ei in range(max_length):
                _, encoder_hidden = encoder(
                    text[:, ei], encoder_hidden)
            encoder_hidden = encoder_hidden.view(audio.size(0), hidden_size)

            audio_output = decoder(encoder_hidden)
            err_g = mse(audio_output, audio)

            err_g.backward()
            optimizer_g.step()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                # print(f"Epoch {epoch}, sample {sample_idx} -- errG: {err_g.item()}, errD: {err_d.item()}")
                print(f"Epoch {epoch}, sample {sample_idx} -- errG: {err_g.item()}")
                _, _, text = random.choice(test_dataset)
                _
                sample(encoder, decoder, max_length, text, hidden_size, 'generated', epoch, sample_idx, 16000)

            if sample_idx % cfg.get('save_iter', 1000) == 0:
                with open('audio_encoder_2.pt', 'wb') as enc_file:
                    torch.save(encoder.state_dict(), enc_file)

                with open('audio_decoder_2.pt', 'wb') as dec_file:
                    torch.save(decoder.state_dict(), dec_file)


def sample(encoder, decoder, max_length, text, hidden_size, outfolder, epoch, sample_idx, sample_rate):
    with torch.no_grad():
        encoder_hidden = encoder.initHidden()

        for ei in range(max_length):
            _, encoder_hidden = encoder(
                text[:, ei], encoder_hidden)
        encoder_hidden = encoder_hidden.view(text.size(0), hidden_size)

        audio_output = decoder(encoder_hidden)

    torchaudio.save("{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx), audio_output.view(-1).cpu(), sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
