from keras.layers import (
    Conv1D, Input, TimeDistributed, LSTM
)
from keras.models import Model


def build_model():
    """
    Builds model and returns it

    :returns keras.models.Model
    """

    inputs = Input(shape=(None, 100, 1))
    conv1 = TimeDistributed(Conv1D(10, 7))(inputs)
    return Model(inputs, conv1)
