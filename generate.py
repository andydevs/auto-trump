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
args = parser.parse_args()

# Load model and preprocessor
model = tf.keras.models.load_model('files/models/saved-model.h5')
with open('files/support/tokenizer.json', 'r') as jsonf:
    jsons = jsonf.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(jsons)

# Create a random starting word
word = randint(1, len(tokenizer.word_index))

# Predict next words in sequence
sequence = []
for i in range(args.words):
    sequence.append(word)
    word_labels = model.predict([word])
    word = np.argmax(word_labels)
    word = int(word)

# Append to text
text = tokenizer.sequences_to_texts([sequence])
print(text)