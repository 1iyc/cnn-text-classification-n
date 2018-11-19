#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Refining Data & Class File before clustering")

# input file
parser.add_argument('--data_file', dest="data_file", type=str, default="./data/data.txt", help="Data File Path")
parser.add_argument('--class_file', dest="class_file", type=str, default="./data/class.txt", help="Data's Class File Path")

#

args = parser.parse_args()

def progress_count(count, total_count, duplicated_count, iter):
    if count % iter == 0:
        print(count, "/", total_count, round(float(count / total_count * 100), 2), "%", duplicated_count, "duplicated found")

def del_duplicates(data_file, class_file):
    data_examples = list(open(data_file, "r", encoding='utf-8').readlines())
    class_examples = list(open(class_file, "r", encoding='utf-8').readlines())

    total_count = len(data_examples)

    f = open(data_file + "_duplicates_list.txt", 'w')

    count = 0
    duplicated_count = 0
    for data in data_examples:
        count += 1
        duplicated_index = [i for i, v in enumerate(data_examples) if v == data]
        for j in reversed(duplicated_index[1:]):
            count += 1
            duplicated_count += 1
            f.write(data_examples[j].strip() + "\t" + class_examples[j].strip() + "\n")
            del data_examples[j]
            del class_examples[j]
            progress_count(count, total_count, duplicated_count, 1000)
        progress_count(count, total_count, duplicated_count, 1000)

    f.close()

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