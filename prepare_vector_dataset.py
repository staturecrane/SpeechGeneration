from speech_generation.utils.text_utils import load_dataset

dataset = load_dataset('all-data')
dataset_values = dataset.values()

with open('fasttext_dataset.txt', 'w') as datafile:
    for row in dataset_values:
        datafile.write(f'{row}\n')

