import os
import random
import sys
sys.path.append('.')

import torch
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
    audio_model = AudioCNN(hidden_size, 2, 1).to(DEVICE)
    discriminator = Discriminator(2, 1).to(DEVICE)

    params = list(embedding_model.parameters()) + list(audio_model.parameters())
    optimizer_g = torch.optim.Adam(params, lr=0.0001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    mse = torch.nn.MSELoss()
    bce = torch.nn.BCELoss()

    real_label = 1
    fake_label = 0

    one = torch.FloatTensor([1]).to(DEVICE)
    mone = one * -1

    err_d = 0
    err_g = 0
    critic_iters = cfg.get('critic_iters', 5)

    for epoch in range(100):
        # for sample_idx, (text, (sample_rate, audio_target)) in enumerate(dataset):
        for sample_idx in range(len(dataset)):
            text, (sample_rate, audio_target) = random.choice(dataset)
            try:
                audio_target = audio_utils.reshape_audio(audio_target, sample_rate).to(DEVICE)
            except:
                continue

            embedding_model.train()
            audio_model.train()
            discriminator.train()

            embedding_model.zero_grad()
            audio_model.zero_grad()
            discriminator.zero_grad()

            for p in discriminator.parameters():
                p.requires_grad_(False) # freeze D

            char_inputs = text_utils.get_input_vectors(text, DEVICE)

            embeddings = embedding_model.initHidden()
            for c_vector in char_inputs:
                embeddings = embedding_model(c_vector.unsqueeze(0), embeddings)

            audio_output = audio_model(embeddings)

            output = discriminator(audio_output)
            err_g = output.mean().view(1, -1)
            err_g.backward(mone)
            # err_g = bce(output, label)
            err_g = -err_g
            # if sample_idx == 0 or err_d < err_g:
            optimizer_g.step()

            for p in discriminator.parameters():
                p.requires_grad_(True) # unfreeze D

            for j in range(1, critic_iters):
                text, (sample_rate, audio_target) = random.choice(dataset)
                try:
                    audio_target = audio_utils.reshape_audio(audio_target, sample_rate).to(DEVICE)
                except:
                    continue

                with torch.no_grad():
                    char_inputs = text_utils.get_input_vectors(text, DEVICE)

                    embeddings = embedding_model.initHidden()
                    for c_vector in char_inputs:
                        embeddings = embedding_model(c_vector.unsqueeze(0), embeddings)

                    audio_output = audio_model(embeddings)

                d_output_real = discriminator(audio_target.view(1, 1, -1))
                # label = torch.full((1,), real_label, device=DEVICE)
                # d_error_real = bce(d_output_real, label)

                # d_error_real.backward(retain_graph=True)
                d_output_fake = discriminator(audio_output.detach())
                # label.fill_(fake_label)
                # d_error_fake = bce(d_output_fake, label)
                # d_error_fake.backward()

                err_d = d_output_fake - d_output_real
                err_d.backward()
                # if sample_idx == 0 or err_g < err_d:
                optimizer_d.step()
                for param in discriminator.parameters():
                    param.data.clamp_(-0.1, 0.1)

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx}: errG: {err_d.mean().item()}, errD: {err_g.mean().item()}")
                # print(f"Epoch {epoch}, sample {sample_idx}: {loss.mean().item()}")
                sample(embedding_model, audio_model, sample_rate, epoch, sample_idx, cfg.get('out_folder'))


            if sample_idx % cfg.get('save_iter', 1000) == 0:
                with open('discriminator_model.pt', 'wb') as disc_file:
                    torch.save(discriminator.state_dict(), disc_file)

                with open('embedding_model.pt', 'wb') as embed_file:
                    torch.save(embedding_model.state_dict(), embed_file)

                with open('audio_model.pt', 'wb') as audio_file:
                    torch.save(audio_model.state_dict(), audio_file)


def sample(embedding_model, audio_model, sample_rate, epoch, sample_idx, outfolder):
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

