import numpy as np
import re

def load_data(data_file, class_file, char, category_level):
    if char:
        data_examples = list(open(data_file, "r", encoding='utf-8').readlines())
        data_examples = [re.sub(r"\s{2,}", " <SP> ", re.sub(r"", " ", s.strip())).strip() for s in data_examples]
        print("Data Read & Transformed to Char Finished!")

        class_examples = list(open(class_file, "r", encoding='utf-8').readlines())
    if category_level:
        class_examples = [re.sub(r" ", "", s.strip())[:category_level*2] for s in class_examples]
        print("Code Read & Transformed to Class Finished!")
    else:
        class_examples = [re.sub(r" ", "", s.strip()) for s in class_examples]
        print("Code Read & Transformed to Class Finished!")

    return [data_examples, class_examples]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def del_space(raw_list):
    raw_list = [re.sub(r" ", "", raw) for raw in raw_list]
    raw_list = [re.sub(r"<SP>", " ", raw) for raw in raw_list]
    return raw_list