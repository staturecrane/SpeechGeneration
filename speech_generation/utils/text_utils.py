import os
import unicodedata

from keras.utils import to_categorical

from ..config import ALL_LETTERS, N_LETTERS


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