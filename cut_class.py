#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Cut class")

# input file
parser.add_argument('--class_file', dest="class_file", type=str, default="./data/data.txt", help="Class File Path")

# Class length
parser.add_argument('--class_length', dest="class_length", type=int, default=None, help="class length")

args = parser.parse_args()


def csv_parse(class_file, class_length):
    with open(class_file, 'r') as f:
        with open(class_file[:-4] + "_" + str(class_length) + ".txt", 'w') as g:
            for line in f.readlines():
                g.write(line[:class_length] + "\n")


def main():
    csv_parse(args.class_file, args.class_length)


if __name__ == '__main__':
    main()