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
from speech_generation.speech_model.model import Discriminator, EncoderRNN, TextDecoder, TextEncoder, Vgg16
from speech_generation.speech_model.loader import LibriSpeech, LibriSpeechIDX


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

    dataset = LibriSpeechIDX(data_dir, device, max_time=max_time, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = LibriSpeechIDX(data_dir, device, max_time=max_time, max_length=max_length)

    encoder = EncoderRNN(N_LETTERS, hidden_size, device).to(device)
    encoder_weights = torch.load('encoder.pt')
    encoder.load_state_dict(encoder_weights)

    for param in encoder.parameters():
        param.requires_grad = False

    encoder_audio = TextEncoder(num_convolutional_features, 1).to(device)
    decoder = TextDecoder(num_convolutional_features, 1, num_channels).to(device)
    discriminator = Discriminator(num_convolutional_features, 1).to(device)

    embedding = nn.Embedding(2, 10).to(device)
    vgg = Vgg16(requires_grad=False).to(device)

    # optimizer_rnn = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_encoder = torch.optim.Adam(list(encoder_audio.parameters()) + list(embedding.parameters()), lr=learning_rate)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # encoder_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_rnn, step_size=3, gamma=0.1)
    # decoder_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=3, gamma=0.1)

    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCELoss()

    real_label = 1
    fake_label = 0

    gen_loss = 0
    err_d = 0

    for epoch in range(100):
        for sample_idx, (audio, speaker, text) in enumerate(dataloader):
            # final batch causes error -- TODO: find out why
            try:
                audio, speaker, text = audio.to(device), speaker.to(device), text.to(device)
                encoder.train()
                encoder_audio.train()
                decoder.train()

                # optimizer_rnn.zero_grad()
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()

                embedding.zero_grad()
                encoder_audio.zero_grad()
                decoder.zero_grad()

                current_batch_size = audio.size(0)

                encoder_hidden = encoder.initHidden(current_batch_size)
                encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size).to(device)

                for ei in range(max_length):
                    encoder_output, encoder_hidden = encoder(
                        text[:, ei], encoder_hidden)
                    encoder_outputs[:, ei] = encoder_output[:, 0]
                flattened_encoder_outputs = encoder_outputs.view(current_batch_size, 1, -1)

                gender_embeddings = embedding(speaker)
                flattened_gender_embeddings = gender_embeddings.view(current_batch_size, 1, -1)

                encoder_inputs = torch.cat((flattened_encoder_outputs, flattened_gender_embeddings), 2)

                hidden = encoder_audio(encoder_inputs)
                audio_output = decoder(hidden)
                append = torch.zeros(current_batch_size, 1, 150528 - (audio_output.size(2))).to(device).fill_(0.0)

                real_concat = torch.cat((audio, append.detach()), 2)
                gen_concat = torch.cat((audio_output, append.detach()), 2)

                features_real = vgg(real_concat.view(current_batch_size, 3, 224, 224))
                features_fake = vgg(gen_concat.view(current_batch_size, 3, 224, 224))

                reconstruction_loss = l1(features_fake.relu4_3, features_real.relu4_3)

                # reconstruction_loss = l1(audio_output, audio)
                # reconstruction_loss.backward()

                label = torch.full((current_batch_size,), real_label).to(device)

                for param in discriminator.parameters():
                    param.requires_grad = True

                discriminator.zero_grad()
                disc_real = discriminator(audio)
                err_d_real = mse(disc_real, label.view(current_batch_size, 1))
                err_d_real.backward()

                disc_fake = discriminator(audio_output.detach())
                label.fill_(fake_label)
                err_d_fake = mse(disc_fake, label.view(current_batch_size, 1))
                err_d_fake.backward()

                err_d = err_d_real + err_d_fake

                if err_d > 0.1:
                    optimizer_discriminator.step()

                for param in discriminator.parameters():
                    param.requires_grad = False

                disc_gen = discriminator(audio_output)
                label.fill_(real_label)
                gen_loss = mse(disc_gen, label.view(current_batch_size, 1)) + (reconstruction_loss)
                gen_loss.backward()

                # optimizer_rnn.step()
                optimizer_encoder.step()
                optimizer_decoder.step()

                if sample_idx % cfg.get('sample_iter', 100) == 0:
                    print(f"Epoch {epoch}, sample {sample_idx} -- errD: {err_d}, errG: {gen_loss}")
                    _, _, text = random.choice(test_dataset)
                    text = text.to(device)
                    sample(text,
                        encoder,
                        encoder_audio,
                        decoder,
                        generation_output_directory,
                        epoch,
                        sample_idx,
                        16000,
                        device,
                        max_length)
            except:
                continue
            if sample_idx % cfg.get('save_iter', 1000) == 0:
                encoder_output_name = f'{checkpoint_out_directory}/audio_encoder.pt'
                with open(encoder_output_name, 'wb') as enc_file:
                    torch.save(encoder_audio.state_dict(), enc_file)

                decoder_output_name = f'{checkpoint_out_directory}/audio_decoder.pt'
                with open(decoder_output_name, 'wb') as dec_file:
                    torch.save(decoder.state_dict(), dec_file)


def sample(text, encoder_rnn, encoder_audio, decoder, outfolder, epoch, sample_idx, sample_rate, device, max_length):
    encoder_rnn.eval()
    encoder_audio.eval()
    decoder.eval()
    text = text.unsqueeze(0)
    with torch.no_grad():
        encoder_hidden = encoder_rnn.initHidden(1)
        encoder_outputs = torch.zeros(max_length, encoder_rnn.hidden_size).to(device)

        for ei in range(max_length):
            encoder_output, encoder_hidden = encoder_rnn(
                text[:, ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        hidden = encoder_audio(encoder_outputs.view(1, 1, -1))
        audio_output = decoder(hidden)

    audio_file_name = "{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx)
    output_to_cpu = audio_output.view(-1).cpu()
    torchaudio.save(audio_file_name, output_to_cpu, sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
