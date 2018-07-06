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
    batch_size = cfg.get('batch_size', 10)
    max_seq_length = cfg.get('seq_length', 400)

    dataset = LibriSpeech(data_dir, DEVICE, scale_factor=FACTOR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    embedding_model = EmbeddingRNN(batch_size, max_seq_length, config.N_LETTERS, hidden_size, DEVICE).to(DEVICE)
    audio_model = AudioCNN(hidden_size, 64, 1).to(DEVICE)
    discriminator = Discriminator(64, 1).to(DEVICE)

    params = list(embedding_model.parameters()) + list(audio_model.parameters())
    optimizer_g = torch.optim.Adam(params, lr=5e-5)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    mse = torch.nn.MSELoss()
    cel = torch.nn.CrossEntropyLoss()

    real_label = 1
    fake_label = 0

    err_d = 0
    err_g = 0

    for epoch in range(100):
        for sample_idx, (audio, gender, text) in enumerate(dataloader):
            audio, gender, text = audio.to(DEVICE), gender.to(DEVICE), text.to(DEVICE)

            embedding_model.train()
            audio_model.train()
            discriminator.train()

            embedding_model.zero_grad()
            audio_model.zero_grad()
            discriminator.zero_grad()

            for param in discriminator.parameters():
                param.requires_grad = True

            hidden_real, d_output_real = discriminator(audio)

            err_d = cel(d_output_real, gender)
            err_d.backward()
            optimizer_d.step()

            for param in discriminator.parameters():
                param.requires_grad = False

            hidden = embedding_model.initHidden(audio.size(0))
            embeddings, hidden = embedding_model(text, hidden)

            audio_output = audio_model(hidden.view(batch_size, hidden_size))

            hidden_fake, gen_output = discriminator(audio_output)
            err_g = mse(hidden_fake, hidden_real.detach())
            err_g.backward()
            optimizer_g.step()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx}: errG: {err_g.mean().item()}, errD: {err_d.mean().item()}")
                sample(embedding_model, audio_model, 16000, epoch, sample_idx, cfg.get('out_folder'), hidden_size)


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

    text = text_utils.get_input_vectors(text_utils.unicode_to_ascii("I'M SORRY DAVE, I'M AFRAID I CAN'T DO THAT")).to(DEVICE)

    hidden = torch.zeros(1, 1, hidden_size).to(DEVICE)
    embeddings, hidden = embedding_model(text.unsqueeze(0), hidden)
    audio_output = audio_model(hidden.view(1, hidden_size))

    torchaudio.save("{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx), audio_output.view(-1).cpu(), sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
