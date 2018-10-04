'''
Autoencoder for LibriSpeech audio dataset

Richard Herbert, 2018
richard.alan.herbert@gmail.com
https://github.com/staturecrane
'''
import os
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import yaml

from speech_generation.config import N_LETTERS
from speech_generation.speech_model.model import AttnDecoderRNN, Encoder, TextDecoder, EncoderRNN, Vgg16
from speech_generation.speech_model.loader import LibriSpeechVectors


def main(cfg_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    cfg_file = open(os.path.abspath(cfg_path))
    cfg = yaml.load(cfg_file.read())
    data_dir = cfg.get('data')
    test_data_dir = cfg.get('test-data')

    try:
        assert data_dir
        assert test_data_dir
    except AssertionError:
        raise 'Missing training and/or testing data folder in config'

    hidden_size = cfg.get('hidden_size', 128)
    batch_size = cfg.get('batch_size', 10)
    num_convolutional_features = cfg.get('num_convolutional_features', 128)
    num_channels = cfg.get('num_channels', 1)
    max_length = cfg.get('max_length', 32)
    max_time = cfg.get('max_time', 5)
    learning_rate = cfg.get('learning_rate', 1e-4)

    generation_output_directory = cfg.get('generation_output_directory')
    checkpoint_out_directory = os.path.abspath(cfg.get('checkpoint_out_directory'))

    dataset = LibriSpeechVectors(data_dir, device, max_time=max_time, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = LibriSpeechVectors(data_dir, device, max_time=max_time, max_length=max_length)

    encoder = EncoderRNN(300, hidden_size, device).to(device)
    decoder = TextDecoder(num_convolutional_features, hidden_size, num_channels).to(device)

    vgg = Vgg16(requires_grad=False).to(device)

    optimizer_rnn = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    mse = torch.nn.L1Loss()

    for epoch in range(100):
        for sample_idx, (audio, speaker, text) in enumerate(dataloader):
            audio, speaker, text = audio.to(device), speaker.to(device), text.to(device)

            encoder.train()
            decoder.train()

            encoder.zero_grad()
            decoder.zero_grad()

            optimizer_rnn.zero_grad()
            optimizer_d.zero_grad()

            current_batch_size = audio.size(0)

            encoder_hidden = encoder.initHidden(current_batch_size)

            for ei in range(max_length):
                encoder_output, encoder_hidden = encoder(
                    text[:, ei].unsqueeze(0).float(), encoder_hidden
                )
            audio_output = decoder(encoder_output.view(current_batch_size, hidden_size, 1, 1))

            features_real = vgg(audio.view(current_batch_size, 3, 256, 256))
            features_fake_rnn = vgg(audio_output)

            err_g = mse(features_fake_rnn.relu2_2, features_real.relu2_2)
            err_g.backward()

            optimizer_rnn.step()
            optimizer_d.step()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx} -- errG: {err_g.item()}")
                _, _, text = random.choice(test_dataset)
                text = text.to(device)
                sample(text,
                       encoder,
                       decoder,
                       generation_output_directory,
                       epoch,
                       sample_idx,
                       16000,
                       hidden_size,
                       max_length,
                       device)

            if sample_idx % cfg.get('save_iter', 1000) == 0:
                encoder_output_name = f'{checkpoint_out_directory}/audio_rnn_encoder.pt'
                with open(encoder_output_name, 'wb') as enc_file:
                    torch.save(encoder.state_dict(), enc_file)

                decoder_output_name = f'{checkpoint_out_directory}/audio_decoder.pt'
                with open(decoder_output_name, 'wb') as enc_file:
                    torch.save(decoder.state_dict(), enc_file)


def sample(text, encoder_rnn, decoder, outfolder, epoch, sample_idx, sample_rate, hidden_size, max_length, device):
    encoder_rnn.eval()
    decoder.eval()
    text = text.unsqueeze(0)
    with torch.no_grad():
        encoder_hidden = encoder_rnn.initHidden(1)

        for ei in range(max_length):
            encoder_output, encoder_hidden = encoder_rnn(
                text[:, ei].unsqueeze(0).float(), encoder_hidden
            )
        encoder_reshaped = encoder_output.view(1, hidden_size, 1, 1)
        audio_output_rnn = decoder(encoder_reshaped)

    audio_file_name = "{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx)
    output_to_cpu = audio_output_rnn.view(-1).cpu()
    torchaudio.save(audio_file_name, output_to_cpu, sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
