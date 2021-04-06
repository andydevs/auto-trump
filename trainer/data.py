"""
Retrieve and preprocess
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# Relative path of data input file
DATA_FILE = 'files/data/tweets.csv'
TOKENIZER_FILE = 'files/support/tokenizer.json'

def input_data(display_data, batch, repeat, shuffle):
    """
    Retrieve and preprocess data
    """
    # Read data. Filter for tweets from the man himself from 2016 through 2020
    print('Reading data...')
    df = pd.read_csv(DATA_FILE)
    df = df[ (df['isRetweet'] != 't') & (df['date'] >= '2016-01-01') & (df['date'] <= '2020-12-31') ]
    tweets = df['text']

    # Filter out links
    print('Filtering out links...')
    link = re.compile(r'https?://[\w\-\_\%\+]+(?:[\/\.\?\=\&]+[\w\-\_\%\+]+)+')
    tweets = tweets.apply(lambda tweet: link.sub('', tweet))
    tweets = tweets[ tweets.apply(len) > 0 ]

    # Train tokenizer and tokenize texts
    print('Tokenizing...')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=5000,
        filters='!"“”$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts(tweets)
    if tokenizer.num_words:
        num_indeces = tokenizer.num_words + 1
    else:
        num_indeces = len(tokenizer.word_index) + 1
    print('Number of tokens: ', num_indeces)
    print('Saving file...')
    with open(TOKENIZER_FILE, 'w+') as tfile:
        tfile.write(tokenizer.to_json())
    tokenized_tweets = tokenizer.texts_to_sequences(tweets)

    # Create input sequences
    # Each row becomes a subset of a tweet upto a specific word.
    # e.g. 
    #   Many will
    #   Many will disagree
    #   Many will disagree but
    #   Many will disagree but @FoxNews
    #   Many will disagree but @FoxNews is
    #   Many will disagree but @FoxNews is doing
    #   Many will disagree but @FoxNews is doing nothing
    #   etc...
    # For each tweet
    # Shamelessly copied from Tensorflow's NLP Zero to Hero course on YouTube
    tweet_seqs = []
    for tweet in tqdm(tokenized_tweets, desc='Creting n-gram sequences'):
        for i in range(1, len(tweet)):
            n_gram_seq = tweet[:i+1]
            tweet_seqs.append(n_gram_seq)

    # Pad tweet sequences so they're all the same length
    print('Padding sequences...')
    tweet_seqs = tf.keras.preprocessing.sequence.pad_sequences(tweet_seqs)
    tweet_seqs = np.array(tweet_seqs)

    # Now we take the last column and set it as the output.
    # The remaining columns are our input sequences. So, basically
    # we feed the machine a sequence of words and we train it to 
    # predict the next word
    print('Separating output words...')
    outputs = tweet_seqs[:,-1]
    sequences = tweet_seqs[:,:-1]

    # Create dataset. One-hot encode labels. Shuffle, batch and repeat
    print('Creating dataset...')
    dataset = tf.data.Dataset.from_tensor_slices((sequences, outputs))
    dataset = dataset.map(lambda seq, out: (seq, tf.one_hot(out, depth=num_indeces)))
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat(repeat)

    # Display data sample
    if display_data:
        print(dataset)
        for sequence_batch, output_batch in dataset.take(1):
            print(f'Input sequences: {sequence_batch}')
            print(f'Output Words: {output_batch}')

    # Return dataset and number of word tokens
    return dataset, num_indeces