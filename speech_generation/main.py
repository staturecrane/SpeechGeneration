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
from speech_generation.speech_model.model import EmbeddingRNN, AudioCNN, Discriminator
from speech_generation.speech_model.loader import LibriSpeech

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACTOR = 2**31


def main(cfg_path):
    cfg_file = open(os.path.abspath(cfg_path))
    cfg = yaml.load(cfg_file.read())
    data_dir = cfg.get('dataset_dir')

    try:
        assert data_dir
    except AssertionError:
        raise 'No "dataset_dir" found in config'

    hidden_size = cfg.get('hidden_size', 128)

    dataset = LibriSpeech(data_dir, scale_factor=FACTOR)
    embedding_model = EmbeddingRNN(config.N_LETTERS, hidden_size, device=DEVICE).to(DEVICE)
    audio_model = AudioCNN(hidden_size, 16, 1).to(DEVICE)
    discriminator = Discriminator(16, 1).to(DEVICE)

    params = list(embedding_model.parameters()) + list(audio_model.parameters())
    optimizer_g = torch.optim.RMSprop(params, lr=1e-3)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=1e-3)

    mse = torch.nn.MSELoss()
    bce = torch.nn.BCELoss()

    real_label = 1
    fake_label = 0

    err_d = 0
    err_g = 0

    for epoch in range(100):
        for sample_idx in range(len(dataset)):
            embedding_model.train()
            audio_model.train()
            discriminator.train()

            embedding_model.zero_grad()
            audio_model.zero_grad()
            discriminator.zero_grad()

            text, (sample_rate, audio_target) = random.choice(dataset)
            try:
                audio_target = audio_utils.reshape_audio(audio_target, sample_rate).to(DEVICE)
            except Exception as e:
                print(e)
                continue

            for param in discriminator.parameters():
                param.requires_grad = True

            char_inputs = text_utils.get_input_vectors(text, DEVICE)

            embeddings = embedding_model.initHidden()
            for c_vector in char_inputs:
                embeddings = embedding_model(c_vector.unsqueeze(0), embeddings)

            audio_output = audio_model(embeddings)

            real_audio_target = torch.cat((audio_target, audio_target), 0).view(1, 1, -1)

            hidden_real, d_output_real = discriminator(real_audio_target)
            label = torch.full((1,), real_label, device=DEVICE)
            # d_error_real = bce(d_output_real, label)
            d_error_real = 0.5 * torch.mean((d_output_real - label)**2)
            fake_audio_target = torch.cat((audio_output.detach, audio_target.view(1, 1, -1)), 2)
            _, d_output_fake = discriminator(fake_audio_target.detach())

            label = torch.full((1,), fake_label, device=DEVICE)
            # d_error_fake = bce(d_output_fake, label)
            d_error_fake = 0.5 * torch.mean((d_output_fake - label)**2)

            err_d = d_error_real + d_error_fake
            err_d.backward()
            optimizer_d.step()

            for param in discriminator.parameters():
                param.requires_grad = False

            hidden_fake, gen_output = discriminator(fake_audio_target)
            # err_g = mse(hidden_fake, hidden_real.detach())
            label = torch.full((1,), real_label, device=DEVICE)
            err_g = 0.5 * torch.mean((gen_output - label)**2)
            err_g.backward()
            optimizer_g.step()

            # output, _ = discriminator(audio_output)
            # label = torch.full((1,), real_label, device=DEVICE)
            # err_g = mse(output, label)
            # err_g.backward()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx}: errG: {err_g.mean().item()}, errD: {err_d.mean().item()}")
                sample(embedding_model, audio_model, sample_rate, epoch, sample_idx, cfg.get('out_folder'), hidden_size)


            if sample_idx % cfg.get('save_iter', 1000) == 0:
                with open('discriminator_model.pt', 'wb') as disc_file:
                    torch.save(discriminator.state_dict(), disc_file)

                with open('embedding_model.pt', 'wb') as embed_file:
                    torch.save(embedding_model.state_dict(), embed_file)

                with open('audio_model.pt', 'wb') as audio_file:
                    torch.save(audio_model.state_dict(), audio_file)


def sample(embedding_model, audio_model, sample_rate, epoch, sample_idx, outfolder, hidden_size):
    embedding_model.eval()
    audio_model.eval()

    text = text_utils.get_input_vectors(text_utils.unicode_to_ascii("I'M AFRAID I CAN'T DO THAT DAVE"),
                                        device=DEVICE)
    embeddings = embedding_model.initHidden()
    for c_vector in text:
        embeddings = embedding_model(c_vector.unsqueeze(0), embeddings)

    audio_output = audio_model(embeddings)
    torchaudio.save("{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx), audio_output.view(-1).cpu(), sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
