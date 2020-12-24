import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def input_fn(buffer_size=3, batch_size=16, data_path='.', min_sentence_length=5, dataset_fraction=1,
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
    num_files = len(text_files)
    random.shuffle(text_files)
    text_files = text_files[:int(num_files * dataset_fraction)]

    texts = tf.data.TextLineDataset(text_files)
    tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocabulary_size)
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
                                yield tf.convert_to_tensor([" ".join(word_buffer)]), tf.convert_to_tensor([line[i]])
                                word_buffer.append(line[i])
            except Exception as e:
                print(f'Failed to open text file {text}')
                print(e)

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))
    dataset = dataset.batch(batch_size).shuffle(1000)

    return dataset, tokenizer
