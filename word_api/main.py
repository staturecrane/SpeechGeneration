"""
Main runtime for FunnelAI NLP service
"""
import json
import os
import re

from flask import Flask, jsonify, request, Response
from gensim.models import FastText

APP = Flask(__name__)

if os.getenv('ENVIRONMENT') != 'TEST':
    WORD_VECTORS = FastText.load_fasttext_format('librispeech')


def get_word_vectors(word_vectors, input_text):
    # input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
    # sentences_strings = []
    # for line in input_text_noparens.split('\n'):
    #     m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    #     sentences_strings.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
    # sentences = []
    # for sent_str in sentences_strings:
    #     tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    #     sentences.append(tokens)
    # results = []
    # for sentence in sentences:
    #     for word in sentence:
    #         try:
    #             results.append(word_vectors[word].tolist())
    #         except KeyError:
    #             print(f'Word {word} not in vector dictionary')
    #             continue
    results = []
    for word in input_text.split(' '):
        try:
            results.append(word_vectors[word].tolist())
        except KeyError:
            print(f'Wored {word} not in dictionary')
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


if __name__ == '__main__':
    APP.run(debug=True)
