from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('twitter.word2vec.txt')
word_vectors.save_word2vec_format('twitter.word2vec.bin', binary=True)
