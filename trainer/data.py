"""
Retrieve and preprocess
"""
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

# Relative path of data input file
MAX_TOKENS = 5000
DATA_FILE = 'files/data/tweets.csv'
TOKENIZER_FILE = 'files/support/tokenizer.json'

# Start and end date
START_DATE = '2016-01-01'
END_DATE = '2020-12-31'


def input_data(display_data, train_frac, batch, repeat, shuffle):
    """
    Retrieve and preprocess data
    """
    # Read data. Filter for tweets from the man himself within start and end dates
    print('Reading data...')
    df = pd.read_csv(DATA_FILE)
    df = df[ (df['isRetweet'] != 't') & (df['date'] >= START_DATE) & (df['date'] <= END_DATE) ]
    tweets = df['text']

    # String encode
    print('Encoding...')
    tweet_chars = tf.strings.unicode_split(tweets, input_encoding='UTF-8')
    encode_chars = tf.keras.layers.StringLookup()
    encode_chars.adapt(tweet_chars)
    tweet_char_ids = encode_chars(tweet_chars)
    vocab_size = encode_chars.vocab_size()

    # Create input sequences
    # Each row becomes a subset of a tweet upto a specific word.
    # e.g.
    #   M
    #   Ma
    #   Man
    #   Many
    #   Many 
    #   Many w
    #   etc...
    # For each tweet
    # Shamelessly copied from Tensorflow's NLP Zero to Hero course on YouTube
    tweet_seqs = []
    for tweet in tqdm(tweet_char_ids, desc='Creating n-gram sequences', total=tweet_char_ids.shape[0]):
        ngrams = [ tweet[:i+1] for i in range(1,len(tweet)) ]
        tweet_seqs.extend(ngrams)

    # Pad tweet sequences so they're all the same length
    print('Padding sequences...')
    tweet_seqs = tf.keras.preprocessing.sequence.pad_sequences(tweet_seqs, 
        maxlen=280, 
        padding='pre',
        truncating='pre')

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
    print('Split into training and testing...')
    train_num = int(train_frac*len(dataset))
    train_dataset = dataset.take(train_num)
    test_dataset = dataset.skip(train_num)

    # Return dataset and number of word tokens
    return train_dataset, test_dataset