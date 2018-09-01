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
from speech_generation.speech_model.model import EncoderRNN, AttnDecoderRNN
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

    test_dataset = LibriSpeech(test_data_dir, DEVICE, scale_factor=FACTOR, max_time=max_time, max_length=max_length)

    encoder = EncoderRNN(config.N_LETTERS, hidden_size, DEVICE).to(DEVICE)
    decoder = AttnDecoderRNN(hidden_size, config.N_LETTERS, dropout_p=0.1).to(DEVICE)

    optimizer_g = torch.optim.Adam(encoder.parameters() + decoder.parameters(), lr=2e-4)

    criterion = torch.nn.NLLLoss()

    for epoch in range(100):
        for sample_idx, (audio, speaker, text) in enumerate(dataloader):
            audio, speaker, text = audio.to(DEVICE), speaker.to(DEVICE), text.to(DEVICE)
            encoder.zero_grad()
            decoder.zero_grad()
            optimizer_g.zero_grad()

            batch_size = audio.size(0)

            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

            for ei in range(max_length):
                encoder_output, encoder_hidden = encoder(
                    text[:, ei], encoder_hidden
                )
                encoder_outputs[ei] = encoder_output[0, 0]


            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            loss = 0
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, text[di])
                    decoder_input = target_tensor[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(decoder_output, target_tensor[di])

            loss.backward()
            optimizer_g.step()
            if sample_idx % cfg.get('sample_iter', 100) == 0:
                print(f"Epoch {epoch}, sample {sample_idx} -- errG: {err_g.item()}, errD: {0}")
                _, speaker, text = random.choice(test_dataset)
                text = text.to(DEVICE)
                speaker = speaker.to(DEVICE)
                sample(text, speaker, encoder, encoder_downsample, leaky_relu, batch_norm, decoder, 'generated', epoch, sample_idx, 16000, max_length, batch_size, hidden_size)

            if sample_idx % cfg.get('save_iter', 1000) == 0:
                with open('encoder.pt', 'wb') as enc_file:
                    torch.save(encoder.state_dict(), enc_file)

                with open('audio_decoder.pt', 'wb') as dec_file:
                    torch.save(decoder.state_dict(), dec_file)



def sample(text, speaker, encoder, encoder_downsample, leaky_relu, batch_norm, decoder, outfolder, epoch, sample_idx, sample_rate, max_length, batch_size, hidden_size):
    with torch.no_grad():
        """
        text = text_utils.get_input_vectors(
            text_utils.unicode_to_ascii("I'M SORRY DAVE I'M AFRAID I CAN'T DO THAT")
        ).to(DEVICE)
        """
        text = text.unsqueeze(0)
        speaker = speaker.unsqueeze(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(max_length):
            encoder_output, encoder_hidden = encoder(
                text[:, ei], encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        encoder_outputs = encoder_outputs.view(batch_size, max_length * hidden_size)
        downsampled = encoder_downsample(encoder_outputs)
        downsampled = downsampled.unsqueeze(1)
        downsampled = batch_norm(downsampled)
        downsampled = leaky_relu(downsampled)
        downsampled = downsampled.view(batch_size, 4, 2352)

        audio_output = decoder(downsampled)

    torchaudio.save("{}/{:06d}-{:06d}.wav".format(outfolder, epoch, sample_idx),
                    audio_output.view(-1).cpu(), sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
