import os
import random
from collections import deque

import tensorflow as tf


def input_fn(buffer_size, data_path, min_sentence_length, dataset_fraction):
    """

    :return: tuple of n previous words, and next word
    """

    # create generator for tensorflow dataset
    text_files = os.listdir(data_path)
    text_files = [os.path.join(data_path, f) for f in text_files]
    num_files = len(text_files)
    random.shuffle(text_files)
    text_files = text_files[:int(num_files * dataset_fraction)]

    texts = tf.data.TextLineDataset(text_files)
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10000)
    encoder.adapt(texts)

    def generator():
        for text_file in text_files:
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

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))

    return dataset, encoder


if __name__ == '__main__':
    dataset, encoder = input_fn(3, os.path.join('..', '..', 'ANC_training_data'), 10)
    # dataset = dataset.shuffle(buffer_size=10000)
    # start_time = time.time()
    dataset_numpy = dataset.as_numpy_iterator()
    # print(len(list(dataset_numpy)))
    # print("--- %s seconds ---" % (time.time() - start_time))
    vocab_iter = iter(encoder.get_vocabulary())
    while not input() == 'q':
        # next_example = next(dataset_numpy)
        # print(next_example)
        print(next(vocab_iter))


def create_keras_model():
    pass
