import numpy as np

import tensorflow as tf
import os


def input_fn(buffer_size, data_path, min_sentence_length):
    """

    :return: tuple of n previous words, and next word
    """

    # create generator for tensorflow dataset
    def generator():
        from collections import deque

        for text_file in os.listdir(data_path):
            word_buffer = deque(maxlen=buffer_size)
            with open(os.path.join(data_path, text_file), encoding='utf-8') as text:
                lines = text.readlines()
                for line_str in lines:
                    line = line_str.split()
                    if len(line) > min_sentence_length:
                        word_buffer.append(line[0])
                        for i in range(1, len(line)):
                            yield tf.stack(word_buffer), line[i]
                            word_buffer.append(line[i])

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=(tf.string, tf.string),
                                             # output_shapes=([None], 1)
                                             )
    return dataset


if __name__ == '__main__':
    dataset = input_fn(3, os.path.join('..', '..', 'ANC_training_data'), 10)
    import time
    start_time = time.time()
    print(len(list(dataset.as_numpy_iterator())))
    print("--- %s seconds ---" % (time.time() - start_time))


def create_keras_model():
    pass
