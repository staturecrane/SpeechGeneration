from keras.layers import (
    Embedding, Input, TimeDistributed, LSTM,
    Bidirectional, Dense
)
from keras.models import Model

from speech_generation.config import N_LETTERS, MAX_INPUT_LENGTH


def build_embedding_model():
    """
    Builds model and returns it

    :returns keras.models.Model
    """

    inputs = Input(shape=(None, N_LETTERS))
    lstm_two = Bidirectional(LSTM(128, recurrent_dropout=0.2))(inputs)
    output = Dense(100)(lstm_two)
    return Model(inputs=inputs, outputs=output)
