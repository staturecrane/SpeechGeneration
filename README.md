# SpeechGeneration

Generating human-like speech using the LibriSpeech dataset and deep-learning libraries.

## Installation 

Requires PyTorch 0.4. Find installation for your OS and Python version [here](https://www.pytorch.org). Then:

```shell
# inside Python3 virtual environment
pip install -r requirements.txt
```
## Preparing Data

Download the "train-clean" dataset (http://www.openslr.org/resources/12/train-clean-360.tar.gz) and the "test-clean" dataset (http://www.openslr.org/resources/12/test-clean.tar.gz). Uncompress them anywhere wherever you wish.

Then run `preprocess_dataset.sh` on each dataset, passing in the absolute path to the sub-folder containing the samples and the output directory for the converted wav and text files. For instance:

```shell
sh preprocess_dataset.sh PATH/TO/LIBRISPEECH/train-clean-360/ PATH/TO/OUTPUT/DIR
sh preprocess_dataset.sh PATH/TO/LIBRISPEECH/test-clean/ PATH/TO/OUTPUT/DIR
```
The samples should now be in the correct format for training.

## Testing

Run tests fist to ensure your dependencies are installed correctly.

```
PYTHONPATH=. pytest speech_generation/tests/
```

## Run Stage One Autoencoder
```
PYTHONPATH=. python speech_generation/stage_one_autoencoder.py config/config.yml
```

## Run Text-to-Speech Generation

**TODO**
