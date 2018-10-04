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
from speech_generation.speech_model.model import AttnDecoderRNN, Encoder, Decoder, EncoderRNN, Vgg16
from speech_generation.speech_model.loader import LibriSpeech


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
    embed_dim = cfg.get('embed_dim', 300)
    max_length = cfg.get('max_length', 32)
    max_time = cfg.get('max_time', 5)
    learning_rate = cfg.get('learning_rate', 1e-4)

    generation_output_directory = cfg.get('generation_output_directory')
    checkpoint_out_directory = os.path.abspath(cfg.get('checkpoint_out_directory'))

    dataset = LibriSpeech(data_dir, device, max_time=max_time, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = LibriSpeech(data_dir, device, max_time=max_time, max_length=max_length)

    encoder = EncoderRNN(N_LETTERS, hidden_size, device).to(device)
    decoder = Decoder(num_convolutional_features, num_channels).to(device)

    encoder_audio = Encoder(num_convolutional_features, num_channels).to(device)

    if cfg.get('checkpoint'):
        checkpoint_directory = os.path.abspath(cfg.get('checkpoint_directory'))

        encoder_weights = torch.load(f'{checkpoint_directory}/audio_encoder.pt')
        encoder_audio.load_state_dict(encoder_weights)

        # decoder_weights = torch.load(f'{checkpoint_directory}/audio_decoder.pt')
        # decoder.load_state_dict(decoder_weights)

    vgg = Vgg16(requires_grad=False).to(device)
    parameters_g = encoder.parameters()
    parameters_d = decoder.parameters()
    optimizer_g = torch.optim.RMSprop(parameters_g, lr=learning_rate)
    optimizer_d = torch.optim.RMSprop(parameters_d, lr=learning_rate)

    mse = torch.nn.L1Loss()

    for epoch in range(100):
        for sample_idx, (audio, speaker, text) in enumerate(dataloader):
            audio, speaker, text = audio.to(device), speaker.to(device), text.to(device)
            decoder.train()
            decoder.zero_grad()
            optimizer_g.zero_grad()

            current_batch_size = audio.size(0)

            encoder_outputs = torch.zeros(max_length, hidden_size, device=device)
            encoder_hidden = encoder.initHidden()

            for ei in range(max_length):
                encoder_output, encoder_hidden = encoder(
                    text[:, ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
            flattened = encoder_outputs.view(1, 4, 2352)

            audio_output = decoder(flattened)

            features_real = vgg(audio.view(current_batch_size, 3, 224, 224))
            features_fake = vgg(audio_output.view(current_batch_size, 3, 224, 224))

            err_g = mse(features_fake.relu2_2, features_real.relu2_2)
            err_g.backward()
            optimizer_g.step()
            optimizer_d.step()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx} -- errG: {err_g.item()}")
                audio_file_name = "{}/{:06d}-{:06d}.wav".format(generation_output_directory, epoch, sample_idx)
                output_to_cpu = audio_output.view(-1).cpu()
                torchaudio.save(audio_file_name, output_to_cpu, 16000)
            #     audio, _, _ = random.choice(test_dataset)
            #     audio = audio.to(device)
            #     sample(audio,
            #            encoder,
            #            decoder,
            #            generation_output_directory,
            #            epoch,
            #            sample_idx,
            #            16000)

            # if sample_idx % cfg.get('save_iter', 1000) == 0:
            #     encoder_output_name = f'{checkpoint_out_directory}/audio_encoder.pt'
            #     with open(encoder_output_name, 'wb') as enc_file:
            #         torch.save(encoder.state_dict(), enc_file)

            #     decoder_output_name = f'{checkpoint_out_directory}/audio-decoder.pt'
            #     with open(decoder_output_name, 'wb') as dec_file:
            #         torch.save(decoder.state_dict(), dec_file)


def sample(audio, encoder, decoder, outfolder, epoch, sample_idx, sample_rate):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoder_hidden = encoder(audio.unsqueeze(0))
        audio_output = decoder(encoder_hidden)

    audio_file_name = "{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx)
    output_to_cpu = audio_output.view(-1).cpu()
    torchaudio.save(audio_file_name, output_to_cpu, sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
