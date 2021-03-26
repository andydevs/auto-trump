"""
Do a dry run of the AI model
"""
import tensorflow as tf
import numpy as np
from random import randint
from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--words', dest='words', default=20)
parser.add_argument('--sequences', dest='sequences', default=10)
args = parser.parse_args()

# Load model and preprocessor
model = tf.keras.models.load_model('files/models/saved-model.h5')
with open('files/support/tokenizer.json', 'r') as jsonf:
    jsons = jsonf.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(jsons)

# Create a random starting word for each sentence
sequences = np.array([
    [randint(1, len(tokenizer.word_index))]
    for i in range(args.sequences)
])

# Predict next words in sequences
for i in range(args.words):
    model.reset_states()
    word_labels = model.predict(sequences)
    words = np.argmax(word_labels, axis=1).reshape(-1,1)
    sequences = np.concatenate((sequences, words), axis=1)

# Append to text
texts = tokenizer.sequences_to_texts(sequences)
print(*texts, sep='\n', end='')