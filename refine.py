#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Refining Data & Class File before clustering")

# input file
parser.add_argument('--data_file', dest="data_file", type=str, default="./data/data.txt", help="Data File Path")
parser.add_argument('--class_file', dest="class_file", type=str, default="./data/class.txt", help="Data's Class File Path")

#

args = parser.parse_args()

def del_duplicates(data_file, class_file):
    print(data_file)
    print(class_file)

def main():
    del_duplicates(args.data_file, args.class_file)

if __name__ == '__main__':
    main()