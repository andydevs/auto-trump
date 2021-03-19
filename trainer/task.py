"""
Run training task
"""
from argparse import ArgumentParser
from .data import input_data

DEF_BATCH = 20
DEF_REPEAT = 5
DEF_SHUFFLE = 200

if __name__ == '__main__':
    # Parse args
    parser = ArgumentParser(description='Run training task')
    parser.add_argument('--display-data', dest='display_data', action='store_true')
    parser.add_argument('--batch', dest='batch', type=int, default=DEF_BATCH)
    parser.add_argument('--repeat', dest='repeat', type=int, default=DEF_REPEAT)
    parser.add_argument('--shuffle', dest='shuffle', type=int, default=DEF_SHUFFLE)
    args = parser.parse_args()

    # Retrieve data
    dataset, num_words = input_data(
        display_data=args.display_data,
        batch=args.batch,
        repeat=args.repeat,
        shuffle=args.shuffle)