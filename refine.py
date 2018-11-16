#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Refining Data & Class File before clustering")

# input file
parser.add_argument('--data_file', dest="data_file", type=str, default="./data/data.txt", help="Data File Path")
parser.add_argument('--class_file', dest="class_file", type=str, default="./data/class.txt", help="Data's Class File Path")

#

args = parser.parse_args()

def del_duplicates(data_file, class_file):
    data_examples = list(open(data_file, "r", encoding='utf-8').readlines())
    class_examples = list(open(class_file, "r", encoding='utf-8').readlines())

    total_count = len(data_examples)

    f = open(data_file + "_duplicates_list.txt", 'w')
    g = open(class_file + "_duplicates_list.txt", 'w')

    result = []
    for i, x in enumerate(data_examples):
        if x in data_examples[:i]:
            f.write(str(i) + "\t" + data_examples[i])
            g.write(str(i) + "\t" + class_examples[i])
            result.append(i)
            del data_examples[i]
            del class_examples[i]
        if (i + 1) % 1000 == 0:
            print((i + 1), "/", total_count, round(float((i+1) / total_count * 100), 2), "%", len(result), "duplicates found")

    f.close()
    g.close()

    with open(data_file + "_refined.txt", 'w') as f:
        for item in data_examples:
            f.write(item)

    with open(class_file + "_refined.txt", 'w') as f:
        for item in class_examples:
            f.write(item)

def main():
    del_duplicates(args.data_file, args.class_file)

if __name__ == '__main__':
    main()