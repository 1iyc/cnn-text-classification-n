#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Refining Data & Class File before clustering")

# input file
parser.add_argument('--data_file', dest="data_file", type=str, default="./data/data.txt", help="Data File Path")

#

args = parser.parse_args()

def csv_parse(data_file):
    with open(data_file, 'r', encoding="utf-8") as f:
        with open(data_file + "_csv_parsed_data.txt", 'w', encoding="utf-8") as g:
            with open(data_file + "_csv_parsed_class.txt", 'w', encoding="utf-8") as h:
                for line in f.readlines():
                    parsed_line_array = line.split(",")
                    g.write("".join(parsed_line_array[1:-2]) + "\n")
                    h.write(parsed_line_array[-2:][0] + "\n")

def main():
    csv_parse(args.data_file)

if __name__ == '__main__':
    main()