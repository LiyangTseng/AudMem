import warnings
warnings.filterwarnings("ignore")
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument('-p', '--preprocess', help='preprocessingi or not', default=False, type=bool)
    args = parser.parse_args()