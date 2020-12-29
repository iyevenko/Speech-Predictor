import os
import re
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import string_ops
from google.cloud import storage

def input_fn(buffer_size=3, batch_size=64, data_path='.', min_sentence_length=5, num_batches=5000,
             vocabulary_size=1000):

    dataset = tf.data.TextLineDataset(data_path)

    tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocabulary_size,
                                                                 output_sequence_length=buffer_size)
    tokenizer.adapt(dataset)

    dataset = dataset.map(tf.strings.strip)
    dataset = dataset.map(tf.strings.lower)
    DEFAULT_STRIP_REGEX = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
    dataset = dataset.map(lambda s: string_ops.regex_replace(s, DEFAULT_STRIP_REGEX, ""))
    dataset = dataset.map(tf.strings.split)
    dataset = dataset.filter(lambda s: tf.shape(s)[0] >= min_sentence_length)
    prev_words = dataset.map(lambda x: x[:min_sentence_length-1])
    prev_words = prev_words.map(lambda s: tf.strings.reduce_join(s, separator=" "))
    next_word = dataset.map(lambda x: x[min_sentence_length-1])
    next_word = next_word.map(lambda s: tf.expand_dims(s, 0))
    next_word = next_word.map(lambda s: tokenizer(s)[0, 0])
    dataset = tf.data.Dataset.zip((prev_words, next_word))
    dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)

    dataset_length = 0
    for _ in dataset.__iter__():
        dataset_length += 1
    print('done bfb counting')

    train_size = int(0.8 * dataset_length)
    val_size = int(0.1 * dataset_length)
    test_size = int(0.1 * dataset_length)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    data_splits = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    return data_splits, tokenizer


def gen_input_fn(buffer_size=3, batch_size=64, data_path='.', min_sentence_length=5, num_batches=5000,
             vocabulary_size=1000, cloud=False):
    """
    :param buffer_size: maximum number of words used to predict next word
    :param batch_size:
    :param data_path: relative path to folder of .txt files used as training data
    :param min_sentence_length: minimum length of sentence to predict next word
    :param dataset_fraction: percentage of dataset to include in training
    :param vocabulary_size: number of words in vocabulary to train model on
    :return: tuple of n previous words, and next word
    """
    text_files = []

    if cloud:
        # Instantiates a client
        client = storage.Client()
        bucket_name = 'speech-predictor-bucket'
        for blob in client.list_blobs(bucket_name, prefix=data_path):
            text_files.append(blob.download_as_text())
        tokenizer_ds = text_files
    else:
        data_path = os.path.join('..', data_path)
        text_files = os.listdir(data_path)
        text_files = [os.path.join(data_path, f) for f in text_files]
        tokenizer_ds = tf.data.TextLineDataset(text_files)

    tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocabulary_size,
                                                                             output_sequence_length=buffer_size)
    tokenizer.adapt(tokenizer_ds)

    def generator():
        for text_file in text_files:
            if not cloud:
                with open(text_file, encoding='utf-8', errors='replace') as text:
                    lines = text.readlines()
            else:
                lines = text_file.splitlines()

            for line_str in lines:
                word_buffer = deque(maxlen=buffer_size)
                line = line_str.split()
                if len(line) > min_sentence_length:
                    word_buffer.append(line[0])
                    for i in range(1, len(line)):
                        prev_words = tf.convert_to_tensor([" ".join(word_buffer)])
                        # keras model doesn't allow string labels
                        # get rid of padding for labels
                        next_token = tokenizer([line[i]])[:, 0]
                        yield prev_words, next_token
                        word_buffer.append(line[i])

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.int32), output_shapes=((1, ), (1, )))
    dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)

    # dataset_len = 0
    #     # print('counting dataset length')
    #     # for _ in dataset.as_numpy_iterator():
    #     #     dataset_len += 1
    # Rough length of dataset for 1% of dataset

    train_size = int(0.8 * num_batches)
    val_size = int(0.1 * num_batches)
    test_size = int(0.1 * num_batches)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size).take(val_size)
    test_dataset = test_dataset.take(test_size)
    #
    # train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    # val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    # test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    data_splits = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    return data_splits, tokenizer

if __name__ == '__main__':
    dataset = input_fn(min_sentence_length=5, data_path=os.path.join('..', 'data', 'raw_sentences.txt'))
    counter = 1
    for d in dataset.__iter__():
        print(counter)
        counter += 1

#     dataset, tokenizer = input_fn(min_sentence_length=3, data_path='data', num_batches=1000)
#     counter = 0
#     print('start')
#     for data in dataset['train'].__iter__():
#         # print(data[0])
#         print(counter)
#         counter+=1
#     print(counter)
#
#     counter = 0
#     for data in dataset['val'].__iter__():
#         # print(data[0])
#         print(counter)
#         counter+=1
#     print(counter)
#
#     counter = 0
#     for data in dataset['test'].__iter__():
#         # print(data[0])
#         print(counter)
#         counter+=1
#     print(counter)
