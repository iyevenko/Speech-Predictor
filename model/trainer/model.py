import os

import tensorflow as tf

import trainer.dataset


class LSTMModel(tf.keras.Model):
    def __init__(self, tokenizer, alpha=0.005, beta=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.get_vocabulary())
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=128, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(128, return_sequences=False)
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.softmax = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def loss(self, y_true, y_pred):
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        scce_loss = scce(y_true, y_pred)

        pred_token = tf.math.argmax(y_pred, axis=-1)
        pred_embed = self.embedding(pred_token)
        true_embed = self.embedding(y_true)

        recon_loss = tf.keras.losses.mse(true_embed, pred_embed)
        recon_loss = tf.reduce_sum(recon_loss)
        self.add_metric(scce_loss, name='Cross_Entropy')
        self.add_metric(recon_loss, name='Reconstruction_Loss')
        return self.alpha*scce_loss + self.beta*recon_loss

    def call(self, inputs, training=None, mask=None):
        tokens = self.tokenizer(inputs)
        embeddings = self.embedding(tokens)
        X = self.lstm(embeddings)
        X = self.lstm2(X)
        X = self.dense(X)
        X = self.dense(X)
        X = self.dense(X)
        X = self.softmax(X)
        return X


if __name__ == '__main__':
    pass
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # model = LSTMModel(tokenizer)
    # sample_text = "I really hope this works."
    # tokens = model.tokenizer(sample_text)
    # # print(tokens)
    #
    # embed = model.embedding(tokens)
    # # print(embed)
    #
    # sample_text = "You like Kanye? Who was in Paris then?"
    # sample_output = model([[sample_text]])
    #
    # data = dataset.as_numpy_iterator()
    # sample_data = next(data)
    # # one batch of tuples with tensors of strings as its elements
    # prev_words, next_word = sample_data
    # pred = model.predict(prev_words)
    #
    # print(model.loss(next_word, pred))

    # # dataset = dataset.shuffle(buffer_size=10000)
    # # start_time = time.time()
    # dataset_numpy = dataset.as_numpy_iterator()
    # # print(len(list(dataset_numpy)))
    # # print("--- %s seconds ---" % (time.time() - start_time))
    # vocab_iter = iter(encoder.get_vocabulary())
    # sample_string = 'Transformer is awesome.'
    # tokenized_string = tokenizer.encode(sample_string)
    # print('Tokenized string is {}'.format(tokenized_string))
    #
    # original_string = tokenizer.decode(tokenized_string)
    # print('The original string: {}'.format(original_string))

    # while not input() == 'q':
    #     next_example = next(vocab_iter)
    #     print(next_example)
    #     print(encoder(next_example[0]))
    #     # print(next(vocab_iter))
