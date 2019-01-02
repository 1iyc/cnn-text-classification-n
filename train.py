#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import time
import os
import datetime
import data_preprocess
from tensorflow.contrib import learn
from text_cnn import TextCNN

# Parameters
tf.flags.DEFINE_string("data_file", "./data/data.txt", "Input Data File Path")
tf.flags.DEFINE_string("class_file", "./data/class.txt", "Output Class File Path")

tf.flags.DEFINE_boolean("char", False, "Classification by Character not Word")

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Summary
tf.flags.DEFINE_boolean("summary", False, "Allow Summary")

FLAGS = tf.flags.FLAGS


def preprocess():
    # Load Data
    print("Data Preprocess Stage...")
    data_text, class_list = data_preprocess.load_data(FLAGS.data_file, FLAGS.class_file, FLAGS.char)

    # Build Vocabulary
    data_max_length = max([len(s.split(" ")) for s in data_text])
    print("Data Max Length: ", data_max_length)
    data_processor = learn.preprocessing.VocabularyProcessor(data_max_length)
    print("Data Processor Made")
    x = np.array(list(data_processor.fit_transform(data_text)))
    del data_text
    print("Data Transformed to NPArray")

    class_processor = learn.preprocessing.VocabularyProcessor(1)
    print("Class Processor Made")
    y_np = np.array(list(class_processor.fit_transform(class_list)))
    del class_list
    print("Class Transformed to NPArray")
    y_max = np.max(y_np)
    print("Number of Class: ", y_max)

    #y = np.zeros((y_np.shape[0], y_max), dtype=int)
    #print("Zero NPArray for Class Made")
    #y_np = y_np.ravel()
    #y[np.arange(y_np.size), y_np-1] = 1
    #y = tf.one_hot(y_np, y_max)
    #print("One-Hot Encoding for Class Finished")
    #del y_np

    # Randomly shuffle data
    np.random.seed(10)
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_np)))
    shuffle_indices = np.random.permutation(np.arange(len(y_np)))
    shuffle_indices = shuffle_indices[dev_sample_index:]
    #x_shuffled = x[shuffle_indices]
    #del x
    #y_shuffled = y[shuffle_indices]
    #del y

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    x_dev = x[shuffle_indices]
    #np.delete(x, shuffle_indices)
    y_dev = y_np[shuffle_indices]
    #np.delete(y, shuffle_indices)

    #del x_shuffled, y_shuffled

    if (FLAGS.char):
        print("Data Character Size: {:d}".format(len(data_processor.vocabulary_)))
        print("Class List Size: {:d}".format(len(class_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_np), len(y_dev)))

    return x, y_np, data_processor, class_processor, x_dev, y_dev


def train(x_train, y_train, data_processor, class_processor, x_dev, y_dev):
    # Training
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        num_classes = np.max(y_train)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=num_classes,
                vocab_size=len(data_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            if FLAGS.summary:
                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            if FLAGS.summary:
                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Checkpoint information
            checkpoint_info = open(os.path.join(os.path.curdir, "runs", "checkpoint_information.txt"), 'a')
            checkpoint_info.write("###################################################\n")
            checkpoint_info.write("Checkpoint Dir Name: " + timestamp)
            if FLAGS.char:
                checkpoint_info.write("\nCHARACTER")
            else:
                checkpoint_info.write("\nWORD")
            checkpoint_info.write("\nFilters' Size: " + FLAGS.filter_sizes)
            checkpoint_info.write("\nNumber of Trained Data: " + str(len(y_train)))
            checkpoint_info.write("\nNumber of Tested Data: " + str(len(y_dev)))
            checkpoint_info.write("\nNumber of Class: " + str(len(class_processor.vocabulary_)))
            checkpoint_info.write("\nnum_epochs: " + str(FLAGS.num_epochs))
            checkpoint_info.write("\nbatch size: " + str(FLAGS.batch_size))
            checkpoint_info.write("\ndropout_keep_prob: " + str(FLAGS.dropout_keep_prob))
            checkpoint_info.write("\n###################################################\n\n\n")
            checkpoint_info.close()

            # Write vocabulary
            if (FLAGS.char):
                data_processor.save(os.path.join(out_dir, "char_data_voca"))
            else:
                data_processor.save(os.path.join(out_dir, "word_data_voca"))

            class_processor.save(os.path.join(out_dir, "class_voca"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                if FLAGS.summary:
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                else:
                    _, step, loss, accuracy = sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy],
                        feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                if FLAGS.summary:
                    train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                if FLAGS.summary:
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                else:
                    step, loss, accuracy = sess.run(
                        [global_step, cnn.loss, cnn.accuracy],
                        feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_preprocess.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...

        y_dev_onehot = np.zeros((y_dev.shape[0], num_classes), dtype=int)
        y_dev = y_dev.ravel()
        y_dev_onehot[np.arange(y_dev.size), y_dev-1] = 1

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            y = np.zeros((len(y_batch), num_classes), dtype=int)
            y_batch = np.asarray(y_batch)
            y_batch = y_batch.ravel()
            y[np.arange(y_batch.size), y_batch-1] = 1
            train_step(x_batch, y)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                if FLAGS.summary:
                    dev_step(x_dev, y_dev_onehot, writer=dev_summary_writer)
                else:
                    dev_step(x_dev, y_dev_onehot)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        print("\nEvaluation:")
        if FLAGS.summary:
            dev_step(x_dev, y_dev_onehot, writer=dev_summary_writer)
        else:
            dev_step(x_dev, y_dev_onehot)
        print("")
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))



def main(argv=None):
    x_train, y_train, data_processor, class_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, data_processor, class_processor, x_dev, y_dev)


if __name__ == '__main__':
    tf.app.run()
