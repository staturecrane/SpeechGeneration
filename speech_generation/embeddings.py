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
from speech_generation.speech_model.model import EncoderRNN, AttnDecoderRNN
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
    learning_rate = cfg.get('learning_rate', 1e-3)

    dataset = LibriSpeech(data_dir, DEVICE, scale_factor=FACTOR, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    encoder = EncoderRNN(config.N_LETTERS, hidden_size, DEVICE).to(DEVICE)
    decoder = AttnDecoderRNN(hidden_size, config.N_LETTERS, DEVICE, max_length=max_length).to(DEVICE)
    teacher_forcing_ratio = 0.5

    encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.RMSprop(decoder.parameters(), lr=learning_rate)
    
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=3, gamma=0.1)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=3, gamma=0.1)

    criterion = torch.nn.NLLLoss()

    for epoch in range(100):
        for sample_idx, (_, _, text) in enumerate(dataloader):
            text = text.to(DEVICE)

            encoder.train()
            decoder.train()

            encoder.zero_grad()
            decoder.zero_grad()

            encoder_hidden = encoder.initHidden()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

            loss = 0

            for ei in range(max_length):
                encoder_output, encoder_hidden = encoder(
                    text[:, ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=DEVICE)

            decoder_hidden = encoder_hidden

            use_teacher_forcing = False # True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    _, target_topi = text[:, di].topk(1)
                    loss += criterion(decoder_output, target_topi[0])
                    decoder_input = target_topi  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()  # detach from history as input

                    _, target_topi = text[:, di].topk(1)
                    loss += criterion(decoder_output, target_topi[0])

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx}: {loss.item()}")
                sample(encoder, decoder, max_length)

            if sample_idx % cfg.get('save_iter', 1000) == 0:
                with open('encoder.pt', 'wb') as embed_file:
                    torch.save(encoder.state_dict(), embed_file)

        encoder_scheduler.step()
        decoder_scheduler.step()

def sample(encoder, decoder, max_length):
    with torch.no_grad():
        text = text_utils.get_input_vectors(text_utils.unicode_to_ascii("I'M SORRY DAVE I'M AFRAID I CAN'T DO THAT")).to(DEVICE)
        text = text.unsqueeze(0)

        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(max_length):
            encoder_output, encoder_hidden = encoder(text[:, ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            decoded_words.append(config.ALL_LETTERS[topi[0][0]])
    
    print(' '.join(decoded_words))

if __name__ == '__main__':
    main(sys.argv[1])
