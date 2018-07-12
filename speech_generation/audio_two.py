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
from speech_generation.speech_model.model import EncoderRNN, AudioCNN, Discriminator
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

    hidden_size = cfg.get('hidden_size', 128)
    batch_size = cfg.get('batch_size', 10)
    max_length = cfg.get('max_length', 400)
    max_time = cfg.get('max_time', 5)
    learning_rate = cfg.get('learning_rate', 1e-3)

    dataset = LibriSpeech(data_dir, DEVICE, scale_factor=FACTOR, max_time=max_time, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    encoder = EncoderRNN(config.N_LETTERS, hidden_size, DEVICE).to(DEVICE)
    encoded_state = torch.load('encoder.pt')
    encoder.load_state_dict(encoded_state)

    for param in encoder.parameters():
        param.requires_grad = False

    discriminator = Discriminator(64, 1).to(DEVICE)
    audio_model = AudioCNN(hidden_size, 64, 1).to(DEVICE)

    optimizer_g = torch.optim.Adam(audio_model.parameters(), lr=1e-4)
    optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=1e-4)

    bce = torch.nn.BCELoss()

    real_label = 1
    fake_label = 0

    err_g = 0
    err_d = 0

    for epoch in range(100):
        for sample_idx, (audio, gender, text) in enumerate(dataloader):
            audio, gender, text = audio.to(DEVICE), gender.to(DEVICE), text.to(DEVICE)

            audio_model.zero_grad()
            optimizer_g.zero_grad()

            discriminator.zero_grad()
            optimizer_d.zero_grad()

            batch_size = audio.size(0)
            """
            encoder_hidden = encoder.initHidden()

            for ei in range(max_length):
                _, encoder_hidden = encoder(
                    text[:, ei], encoder_hidden)
            encoder_hidden = encoder_hidden.view(audio.size(0), hidden_size)
            """
            embeddings = torch.FloatTensor(audio.size(0), hidden_size).to(DEVICE).normal_(0, 1)
            audio_output = audio_model(embeddings)

            for param in discriminator.parameters():
                param.requires_grad = True

            label = torch.full((batch_size,), real_label, device=DEVICE)

            # real_input = torch.cat((audio, audio), 2)

            _, d_output_real = discriminator(audio)
            err_d_real = bce(d_output_real.view(batch_size, 1), label)
            err_d_real.backward()

            # fake_input = torch.cat((audio_output.detach(), audio), 2)
            label = torch.full((batch_size,), fake_label, device=DEVICE)
            _, d_output_fake = discriminator(audio_output.detach())
            err_d_fake = bce(d_output_fake.view(batch_size, 1), label)
            err_d_fake.backward()

            optimizer_d.step()

            err_d = err_d_real + err_d_fake

            for param in discriminator.parameters():
                param.requires_grad = False

            label = torch.full((batch_size,), real_label, device=DEVICE)
            # gen_input = torch.cat((audio_output, audio), 2)
            _, gen_output = discriminator(audio_output)
            err_g = bce(gen_output.view(batch_size, 1), label)
            err_g.backward()

            optimizer_g.step()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx} -- errD: {err_d.item()} errG: {err_g.item()}")
                sample(encoder, audio_model, 'generated', epoch, sample_idx, 16000, max_length, hidden_size)

            if sample_idx % cfg.get('save_iter', 1000) == 0:
                with open('audio_model.pt', 'wb') as audio_file:
                    torch.save(audio_model.state_dict(), audio_file)
            
                with open('discriminator.pt', 'wb') as disc_file:
                    torch.save(discriminator.state_dict(), disc_file)



def sample(encoder, audio_model, outfolder, epoch, sample_idx, sample_rate, max_length, hidden_size):
    with torch.no_grad():
        """
        text = text_utils.get_input_vectors(
            text_utils.unicode_to_ascii("I'M SORRY DAVE I'M AFRAID I CAN'T DO THAT")
        ).to(DEVICE)
        text = text.unsqueeze(0)

        encoder_hidden = encoder.initHidden()

        for ei in range(max_length):
            _, encoder_hidden = encoder(text[:, ei], encoder_hidden)
        """
        embeddings = torch.FloatTensor(1, hidden_size).to(DEVICE).normal_(0, 1)

        audio_output = audio_model(embeddings)
    torchaudio.save("{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx), audio_output.view(-1).cpu(), sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
