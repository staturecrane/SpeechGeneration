# SpeechGeneration
[![Build Status](https://travis-ci.org/staturecrane/SpeechGeneration.svg?branch=master)](https://travis-ci.org/staturecrane/SpeechGeneration)
[![Coverage Status](https://coveralls.io/repos/github/staturecrane/SpeechGeneration/badge.svg?branch=master)](https://coveralls.io/github/staturecrane/SpeechGeneration?branch=master)

Generating human-like speech using the LibriSpeech dataset and deep-learning libraries.

## Installation 

Requires PyTorch 0.4. Find installation for your OS and Python version [here](https://www.pytorch.org). Then:

```shell
# inside Python3 virtual environment
pip install -r requirements.txt
```

## Testing

Run tests fist to ensure your dependencies are installed correctly.

```
PYTHONPATH=. pytest speech_generation/tests/