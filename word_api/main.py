"""
Main runtime for FunnelAI NLP service
"""
import json
import os

from flask import Flask, jsonify, request, Response
from gensim.models import KeyedVectors
from raven.contrib.flask import Sentry
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

APP = Flask(__name__)

if os.getenv('ENVIRONMENT') != 'TEST':
    WORD_VECTORS = KeyedVectors.load_word2vec_format('/app/librispeech_gensim.bin', binary=True)

def get_word_vectors(word_vectors, text):
    results = []
    for word in word_tokenize(text):
        try:
            results.append(word_vectors[word.upper().strip()].tolist())
        except KeyError:
            print(f'Word {word} not in vector dictionary')
            continue
    return results


@APP.route('/vectors', methods=['POST'])
def get_vectors():
    try:
        data = request.json
    except Exception:
        return Response('Wrong content headers or bad request'), 400

    if not data:
       return Response('No message data found'), 400

    message = data.get('message')
    if not message:
        return Response('Please include a "message" key in your JSON payload'), 400

    word_vectors = get_word_vectors(WORD_VECTORS, message)
    return jsonify(word_vectors), 200


@APP.route('/health')
def home():
    """
    Healthcheck for ECS
    """
    return 'FunnelAI NLP Word API', 200


if __name__ == '__main__':
    APP.run(debug=True)
