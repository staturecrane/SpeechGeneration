import os
import sys
sys.path.append('.')

import torch
from torch.autograd import Variable
import torchaudio
from tqdm import tqdm
import yaml

from speech_generation import config
from speech_generation.utils import text_utils, audio_utils
from speech_generation.speech_model.model import EmbeddingRNN, AudioRNN
from speech_generation.speech_model.loader import LibriSpeech 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACTOR = 2**31

flatten = lambda l: torch.Tensor([item for sublist in l for item in sublist])


def main(cfg_path): 
    cfg_file = open(os.path.abspath(cfg_path))
    cfg = yaml.load(cfg_file.read())
    data_dir = cfg.get('dataset_dir')

    try:
        assert data_dir
    except AssertionError:
        raise 'No "dataset_dir" found in config'

    hidden_size = cfg.get('hidden_size', 128)
    chunksize = cfg.get('chunksize', 400)

    dataset = LibriSpeech(data_dir, scale_factor=FACTOR)
    embedding_model = EmbeddingRNN(config.N_LETTERS, hidden_size, device=DEVICE).to(DEVICE)
    audio_model = AudioRNN(hidden_size, chunksize, device=DEVICE).to(DEVICE)

    params = list(embedding_model.parameters()) + list(audio_model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    mse = torch.nn.MSELoss()

    for epoch in range(100):
        for sample_idx, (text, (sample_rate, audio)) in enumerate(dataset):
            embedding_model.train()
            audio_model.train()
            
            embedding_model.zero_grad()
            audio_model.zero_grad()

            chunked_audio = audio_utils.chunk_audio(audio, chunksize=chunksize)
            char_inputs = text_utils.get_input_vectors(text, DEVICE)

            embeddings = embedding_model.initHidden()
            for c_vector in char_inputs:
                embeddings = embedding_model(c_vector.unsqueeze(0), embeddings)

            hidden = audio_model.initHidden()
            loss = 0
            for i, chunk in enumerate(chunked_audio):
                chunk = chunk.to(DEVICE).view(-1)
                if len(chunk) < chunksize:
                    new_chunk = torch.zeros(chunksize).to(DEVICE)
                    for i in range(chunksize):
                        try:
                            new_chunk[i] = chunk[i]
                        except IndexError:
                            new_chunk[i] = 0
                    chunk = new_chunk

                chunk = chunk.unsqueeze(0)
                output, stop, hidden = audio_model(embeddings, hidden)

                if i == len(chunk) - 1:
                    stop_target = torch.Tensor([1]).to(DEVICE).unsqueeze(0)
                else:
                    stop_target = torch.Tensor([0]).to(DEVICE).unsqueeze(0)

                loss += mse(output, chunk) + mse(stop, stop_target)

            print(f"Sample {sample_idx}: {loss.mean()}")
            loss.backward()
            optimizer.step()

            if sample_idx % cfg.get('save_iter', 100) == 0:
                sample(embedding_model, audio_model, sample_rate)


def sample(embedding_model, audio_model, sample_rate, max_length=1000):
    embedding_model.eval()
    audio_model.eval()

    text = text_utils.get_input_vectors(text_utils.unicode_to_ascii("I'M AFRAID I CAN'T DO THAT DAVE"),
                                        device=DEVICE)
    embeddings = embedding_model.initHidden()
    for c_vector in text:
        embeddings = embedding_model(c_vector.unsqueeze(0), embeddings)

    hidden = audio_model.initHidden()
    audio_array = []
    for _ in range(max_length):
        output, stop, hidden = audio_model(embeddings, hidden)
        audio_array.append(output.data[0] * FACTOR)
        if stop.data[0] > 0.75:
            break
    audio_array = flatten(audio_array)
    torchaudio.save('testaudio.wav', audio_array, sample_rate)


if __name__ == '__main__':
    main(sys.argv[1])
 