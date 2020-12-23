import os
import random
from collections import deque

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

def input_fn(buffer_size, data_path, min_sentence_length, dataset_fraction, vocabulary_size):
    """
    :param buffer_size: maximum number of words used to predict next word
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
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocabulary_size)
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

    # hub.load(os.path.abspath('preprocess'))
    # print(os.path.abspath('preprocess'))
    # bert_preprocess_model = tf.keras.models.load_model(os.path.join('..' ,'preprocess'))
    # text_test = ['this is such an amazing movie!']
    # text_preprocessed = bert_preprocess_model(text_test)

    # print(f'Keys       : {list(text_preprocessed.keys())}')
    # print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    # print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    # print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    # print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

    # dataset, encoder = input_fn(3, os.path.join('..', '..', 'ANC_training_data'), 10, 0.02, 5000)
    # # dataset = dataset.shuffle(buffer_size=10000)
    # # start_time = time.time()
    # dataset_numpy = dataset.as_numpy_iterator()
    # # print(len(list(dataset_numpy)))
    # # print("--- %s seconds ---" % (time.time() - start_time))
    # vocab_iter = iter(encoder.get_vocabulary())
    # while not input() == 'q':
    #     next_example = next(dataset_numpy)
    #     print(next_example)
    #     print(encoder(next_example[0]))
    #     # print(next(vocab_iter))

    preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', trainable=True)
    bert = hub.KerasLayer('https://tfhub.dev/google/experts/bert/wiki_books/mnli/2')

    # text_test = ["MY NAME IS FLYNN. Don't call me Walt Jr!"]
    # text_preprocessed = preprocess(text_test)
    #
    # print(f'Keys       : {list(text_preprocessed.keys())}')
    # print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    # print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    # print(f'Input Mask : {text_preprocessed["input_mask"][:]}')
    # print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')
    #
    # bert_results = bert(text_preprocessed)
    #
    # print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    # print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    # print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    # print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
    print("Finished loading weights")
    encoding1 = bert(preprocess(['master']))['pooled_output']
    encoding2 = bert(preprocess(['slave']))['pooled_output']
    encoding3 = bert(preprocess(['toys']))['pooled_output']
    print(tf.keras.losses.MSE(encoding1, encoding2))
    print(tf.keras.losses.MSE(encoding1, encoding3))
    print(tf.keras.losses.MSE(encoding2, encoding3))



class BERTModel(tf.keras.Model):
    def __init__(self, text_encoder, buffer_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.embedding = tf.keras.layers.Embedding(input_dim=buffer_size, output_dim=128)
        self.preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2')
        self.bert = hub.load('https://tfhub.dev/google/experts/bert/wiki_books/mnli/2')
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=False)
        self.dense = tf.keras.layers.Dense(768, activation='relu')
        self.softmax = tf.keras.layers.Dense( activation='softmax')


    def call(self, inputs, training=None, mask=None):
        text_preprocessed = self.preprocess(inputs)
        bert_results = self.bert(text_preprocessed)
        pooled_output = bert_results["pooled_output"]
        sequence_output = bert_results["sequence_output"]

        lstm_output = self.lstm(sequence_output)
        lstm_output = self.dense(lstm_output)
        encoded_prediction = tf.keras.layers.Concatenate([lstm_output, pooled_output])
        encoded_prediction = self.dense(encoded_prediction)

        word_probs = self.softmax(encoded_prediction)


        # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        # [x, x, x, x, x, x, x, 0, x, x, x, x ,x, x]



