#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Refining Data & Class File before clustering")

# input file
parser.add_argument('--refined_data_file', dest="refined_data_file", type=str,
                    default="./data/data.txt_refined.txt", help="Refined Data File Path")
parser.add_argument('--refined_class_file', dest="refined_class_file", type=str,
                    default="./data/class.txt_refined.txt", help="Refined Class File Path")
parser.add_argument('--duplicated_file', dest="duplicated_file", type=str,
                    default="./data/data.txt_duplicates_list.txt", help="Duplicated Data File Path")

# action
parser.add_argument('--correct', dest="correct", type=bool,
                    default=False, help="If true Correct Refined Data using Duplicated Data File Statics")

args = parser.parse_args()

def make_statics(refined_data_file, refined_class_file, duplicated_file):
    refined_data = list(open(refined_data_file, 'r', encoding="utf-8").readlines())
    refined_class = list(open(refined_class_file, 'r', encoding="utf-8").readlines())
    duplicated_data = list(open(duplicated_file, 'r', encoding="utf-8").readlines())

    duplicated_index = 0

    if args.correct:
        g = open(refined_class_file + "_corrected.txt", 'w', encoding="utf-8")

    with open(duplicated_file + "_statics.txt", 'w', encoding="utf-8") as f:
        for i in range(len(refined_data)):
            data = dict()
            f.write(refined_data[i].strip() + "\t" + refined_class[i].strip() + "\t\t")
            for j in range(duplicated_index, len(duplicated_data)):
                if refined_data[i].strip() != duplicated_data[duplicated_index].split("\t")[0]:
                    if args.correct:
                        if len(data):
                            g.write(sorted(data, key=data.get, reverse=True)[0] + "\n")
                        else:
                            g.write(refined_class[i].strip() + "\n")
                    for k in sorted(data, key=data.get, reverse=True):
                        f.write(k + "\t" + str(data[k]) + "\t")
                    f.write("\n")
                    break
                else:
                    if duplicated_data[duplicated_index].split("\t")[1].strip() in data:
                        data[duplicated_data[duplicated_index].split("\t")[1].strip()] += 1
                    else:
                        data[duplicated_data[duplicated_index].split("\t")[1].strip()] = 1
                duplicated_index += 1

def main():
    make_statics(args.refined_data_file, args.refined_class_file, args.duplicated_file)

if __name__ == '__main__':
    main()