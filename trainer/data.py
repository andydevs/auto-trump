"""
Retrieve and preprocess
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

# Relative path of data input file
MAX_TOKENS = 5000
DATA_FILE = 'files/data/tweets.csv'
TOKENIZER_FILE = 'files/support/tokenizer.json'

# Start and end date
START_DATE = '2016-01-01'
END_DATE = '2020-12-31'


def starting_words(top=None):
    """
    Retrieve most common starting words 
    """
    # Read data
    print('Reading data...')
    df = pd.read_csv(DATA_FILE)
    df = df[ (df['isRetweet'] != 't') & (df['date'] >= START_DATE) & (df['date'] <= END_DATE) ]
    tweets = df['text']

    # Filter out links
    print('Filtering out links...')
    link = re.compile(r'https?://[\w\-\_\%\+]+(?:[\/\.\?\=\&]+[\w\-\_\%\+]+)+')
    tweets = tweets.apply(lambda tweet: link.sub('', tweet))
    tweets = tweets[ tweets.apply(len) > 0 ]
    
    # Tokenize
    print('Tokenizing...')
    with open(TOKENIZER_FILE, 'r') as jsonf:
        jsons = jsonf.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(jsons)
    tweets = tokenizer.texts_to_sequences(tweets)

    # Get first word in each tweet
    words = [ sequence[0] for sequence in tweets if len(sequence) > 0 ]

    # Organize words by most common to least
    freqs = defaultdict(lambda: 0)
    for word in words:
        freqs[word] += 1
    counts = np.array(sorted(freqs.items(), key=lambda w: w[1], reverse=True))
    words = counts[:,0]
    probs = counts[:,1]

    # Return most n words or all words
    if top is not None and top < len(words):
        words = words[:top]
        probs = probs[:top]

    # Normalize probabilities
    probs = probs / probs.sum()

    # Return words and probabilities
    return words, probs


def input_data(display_data, train_frac, batch, repeat, shuffle):
    """
    Retrieve and preprocess data
    """
    # Read data. Filter for tweets from the man himself within start and end dates
    print('Reading data...')
    df = pd.read_csv(DATA_FILE)
    df = df[ (df['isRetweet'] != 't') & (df['date'] >= START_DATE) & (df['date'] <= END_DATE) ]
    tweets = df['text']

    # Filter out links
    print('Filtering out links...')
    link = re.compile(r'https?://[\w\-\_\%\+]+(?:[\/\.\?\=\&]+[\w\-\_\%\+]+)+')
    tweets = tweets.apply(lambda tweet: link.sub('', tweet))
    tweets = tweets[ tweets.apply(len) > 0 ]

    # Train tokenizer and tokenize texts
    print('Tokenizing...')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=MAX_TOKENS,
        filters='!"“”$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts(tweets)
    if tokenizer.num_words:
        vocab_size = tokenizer.num_words + 1
    else:
        vocab_size = len(tokenizer.word_index) + 1
    print('Number of tokens: ', vocab_size)
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
    dataset = dataset.map(lambda seq, out: (seq, tf.one_hot(out, depth=vocab_size)))
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat(repeat)

    # Display data sample
    if display_data:
        print(dataset)
        for sequence_batch, output_batch in dataset.take(1):
            print(f'Input sequences: {sequence_batch}')
            print(f'Output Words: {output_batch}')

    # Split into training and testing data
    train_num = int(train_frac*len(dataset))
    train_dataset = dataset.take(train_num)
    test_dataset = dataset.skip(train_num)

    # Return dataset and number of word tokens
    return train_dataset, test_dataset, vocab_size