#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_preprocess
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "./data/data.txt", "Input Data File Path")
tf.flags.DEFINE_string("class_file", "./data/class.txt", "Output Class File Path")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_boolean("char", False, "Classification by Character not Word")

# Class Category
tf.flags.DEFINE_integer("category_level", None, "Category Level (1 or 2) cf. if 1, 20 30 40 => 20 // if 2, 20 30 40 => 20 30")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
#print("")

# CHANGE THIS: Load data. Load your own data here
# TODO: Modify Eval_train
if FLAGS.eval_train:
    x_raw, y_raw = data_preprocess.load_data(FLAGS.data_file, FLAGS.class_file, FLAGS.char, FLAGS.category_level)
    class_vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "class_voca")
    class_processor = learn.preprocessing.VocabularyProcessor.restore(class_vocab_path)
    y_test = np.array(list(class_processor.transform(y_raw)))
    y_test = y_test.ravel()
    #y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off.", "what the fuck", "i love you", "hello, ma friend?", "go to hell", "do you want to be killed?"]
    y_test = [1, 0, 0, 1, 1, 0, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "char_data_voca")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
print('t', FLAGS.checkpoint_dir)
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_preprocess.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = np.array([], dtype='int')
        #all_predictions = np.dtype('uint32')

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        all_predictions = all_predictions + 1

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
if FLAGS.char:
    """
    Del Space
    <SP> to Space
    """
    x_raw = data_preprocess.del_space(x_raw)

all_predictions = ["".join(list(class_processor.vocabulary_.reverse(prediction))) for prediction in all_predictions]

predictions_human_readable = np.column_stack((x_raw, y_raw, all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))

with open(out_path, 'w', newline='') as f:
    csv.writer(f).writerow(['good\'s name', 'original code', 'predicted code'])
    csv.writer(f).writerows(predictions_human_readable)
