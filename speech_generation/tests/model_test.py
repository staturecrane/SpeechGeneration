import random

import numpy as np

from speech_generation.speech_model  import build_embedding_model
import speech_generation.utils as utils

TEST_DATA_DIR = 'speech_generation/tests/test_data'


def test_embedding_model():
    model = build_model()

    sample_dataset = utils.get_filenames_and_text(f'{TEST_DATA_DIR}/text1.txt')
    random_sample = random.choice(list(sample_dataset.items()))
    
    input_tensors = np.array([utils.convert_to_onehot(char) for char in random_sample[1]])

    output = model.predict(np.expand_dims(input_tensors, axis=0))
