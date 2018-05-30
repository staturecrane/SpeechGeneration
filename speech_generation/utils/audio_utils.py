from keras.layers import (
    Conv1D, Input, TimeDistributed, LSTM
)
from keras.models import Model
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import os


def load_audio(filepath):
    """
    Loads and returns sample rate and audio for a specific file

    :param filepath: Path to .wav audio file to open

    :returns (sample_rate, audio[])
    """

    audio_binary = tf.read_file(os.path.abspath(filepath))
    desired_channels = 1
    wav_decoder = contrib_audio.decode_wav(audio_binary,
                                           desired_channels=desired_channels)

    with tf.Session() as sess:
        sample_rate, audio = sess.run([
            wav_decoder.sample_rate,
            wav_decoder.audio
        ])

        return sample_rate, audio


def build_model():
    """
    Builds model and returns it

    :returns keras.models.Model
    """

    inputs = Input(shape=(None, 100, 1))
    conv1 = TimeDistributed(Conv1D(10, 7))(inputs)
    return Model(inputs, conv1)


def chunk_audio(audio, chunksize=100):
    return [audio[i:i+chunksize] for i in range(0, len(audio), chunksize)]

