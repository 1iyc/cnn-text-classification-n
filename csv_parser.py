#! /usr/bin/env python

import argparse
import csv

parser = argparse.ArgumentParser(description="Refining Data & Class File before clustering")

# input file
parser.add_argument('--data_file', dest="data_file", type=str, default="./data/data.txt", help="Data File Path")

# extract evaluate rate
parser.add_argument('--eval_rate', dest="eval_rate", type=float, default=None, help="evaluation rate")

args = parser.parse_args()


def csv_parse(data_file, eval_rate):
    if eval_rate:
        import random
        total_count = len(open(data_file).readlines())
        eval_count = int(total_count * eval_rate)
        eval_list = sorted(random.sample(range(total_count), eval_count))
    with open(data_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # skip first line
        next(csv_reader)
        count = 1
        for lines in csv_reader:
            if count in eval_list:
                with open("./eval_class_" + data_file.split("/")[-1][:-4] + ".txt", 'a', encoding="utf-8") as f:
                    f.write(lines[0].strip() + "\n")
                with open("./eval_name_" + data_file.split("/")[-1][:-4] + ".txt", 'a', encoding="utf-8") as f:
                    f.write("".join(lines[1:]).strip() + "\n")
                count += 1
                continue
            else:
                with open("./train_class_" + data_file.split("/")[-1][:-4] + ".txt", 'a', encoding="utf-8") as f:
                    f.write(lines[0].strip() + "\n")
                with open("./train_name_" + data_file.split("/")[-1][:-4] + ".txt", 'a', encoding="utf-8") as f:
                    f.write("".join(lines[1:]).strip() + "\n")
                count += 1
                continue


def main():
    csv_parse(args.data_file, args.eval_rate)


if __name__ == '__main__':
    main()