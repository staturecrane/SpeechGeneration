import sys
from gensim.models import KeyedVectors

filename = sys.argv[1]

word_vectors = KeyedVectors.load_word2vec_format(f'{filename}.vec')
word_vectors.save_word2vec_format(f'{filename}_gensim.bin', binary=True)
