wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip 
python -m gensim.scripts.glove2word2vec --input glove.twitter.27B.200d.txt --output twitter.word2vec.txt
rm -rf glove.twitter.27B*
python convert_vectors.py
rm -rf twitter.word2vec.txt
