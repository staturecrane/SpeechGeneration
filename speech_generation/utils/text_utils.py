import os
import unicodedata

from keras.utils import to_categorical
import numpy as np
import requests
import torch

from speech_generation.config import ALL_LETTERS, N_LETTERS
from speech_generation.utils.utils import merge_dicts


def get_text_files(directory):
    """
    Returns all files with .txt extension from given directory

    Args:
        directory (string): path to directory to parse

    Returns:
        list: list of absolute paths to all found text files
    """
    directory_path = os.path.abspath(directory)
    text_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            text_files.append(os.path.join(directory_path, filename))


    return text_files


def get_filenames_and_text(textfile):
    """
    Splits LibriSpeech dataset files into their filename keys and associate text samples

    Args:
        textfile (string): path to LibriSpeech dataset text file

    Returns:
        dict: with value {filename_key: text_sample}
    """
    with open(os.path.abspath(textfile)) as textfile:
        lines = textfile.readlines()
        return {filename: unicode_to_ascii(sample) for filename, sample in split_lines(lines)}


def load_dataset(data_dir):
    return merge_dicts(*[get_filenames_and_text(txt) for txt in get_text_files(data_dir)])


def split_lines(lines):
    """
    Args:
        lines (list): list of lines from a single LibriSpeech dataset text file

    Yields:
        (string, string): tuple containing the filename key and its associated sample
                          for each given line
    """
    for line in lines:
        sample_splits = line.split(' ')
        yield sample_splits[0], ' '.join(sample_splits[1:]).replace('\n', '')


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def convert_to_onehot(char):
    return to_categorical(ALL_LETTERS.index(char), num_classes=N_LETTERS)


def get_input_vectors(line, max_length=400):
    one_hot = [convert_to_onehot(char) for char in line]
    if len(one_hot) < max_length:
        for i in range(max_length - len(one_hot)):
            one_hot.append(convert_to_onehot(unicode_to_ascii(' ')))
    if len(one_hot) > max_length:
        one_hot = one_hot[:max_length]
    return torch.LongTensor(one_hot)


def get_input_word_vectors(line, max_length=37):
    data = requests.post('http://localhost:8000/vectors', json={'message': line})
    vectors = data.json()
    embed_dim = len(vectors[0])
    vector_length = len(vectors)
    if vector_length < max_length:
        for _ in range(max_length - vector_length):
            vectors.append(np.zeros(embed_dim))
    elif len(vectors) > max_length:
        vectors = vectors[:max_length]
    return np.array(vectors)
