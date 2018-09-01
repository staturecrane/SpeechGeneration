import random

import numpy as np
import pytest
import torch

from speech_generation.speech_model.model import Encoder, Decoder
from speech_generation.utils import text_utils

TEST_DATA_DIR = 'speech_generation/tests/test_data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
HIDDEN_SIZE = 128
OUTPUT_SIZE = 100

# TODO
