import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def input_fn(buffer_size=5, batch_size=64, data_path='.', min_sentence_length=5, num_batches=5000,
             vocabulary_size=10000):
    """
    :param buffer_size: maximum number of words used to predict next word
    :param batch_size:
    :param data_path: relative path to folder of .txt files used as training data
    :param min_sentence_length: minimum length of sentence to predict next word
    :param dataset_fraction: percentage of dataset to include in training
    :param vocabulary_size: number of words in vocabulary to train model on
    :return: tuple of n previous words, and next word
    """
    text_files = os.listdir(data_path)
    text_files = [os.path.join(data_path, f) for f in text_files]

    texts = tf.data.TextLineDataset(text_files)
    tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocabulary_size, output_sequence_length=buffer_size)
    tokenizer.adapt(texts)

    def generator():
        for text_file in text_files:
            try:
                with open(text_file, encoding='utf-8') as text:
                    lines = text.readlines()
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
            except Exception as e:
                print(f'Failed to open text file {text}')
                print(e)

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
