"""
Run training task
"""
from argparse import ArgumentParser
from .data import get_data

if __name__ == '__main__':
    # Parse args
    parser = ArgumentParser(description='Run training task')
    parser.add_argument('--display-data', dest='display_data', action='store_true')
    args = parser.parse_args()

    # Retrieve data
    get_data(display_data=args.display_data)