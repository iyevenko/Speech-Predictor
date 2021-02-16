import datetime
import os
import time

import numpy as np
import tensorflow as tf


def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # mask -> (batch_size, 1, 1, seq_len)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    # mask -> (seq_len, seq_len)
    return mask


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_tar_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_tar_padding_mask, look_ahead_mask)

    return enc_padding_mask, look_ahead_mask, dec_padding_mask


def positional_encoding(position, d_model):
    pos = np.arange(position)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angles = pos * angle_rates

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    pos_encoding = angles[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.softmax = tf.keras.layers.Softmax()

    # noinspection PyMethodOverriding
    def call(self, K, V, Q, mask):
        QK = tf.matmul(Q, K, transpose_b=True)

        dk = tf.cast(self.d_model, dtype=tf.float32)
        logits = QK / tf.math.sqrt(dk)
        
        if mask is not None:
            # add -inf to logits so masked probs are zero
            logits += (mask * -1e9)

        X = self.softmax(logits)
        X = tf.matmul(X, V)

        return X


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # num_heads must by divisor of d_model
        self.head_depth = d_model // num_heads

        self.Wk = tf.keras.layers.Dense(d_model)
        self.Wv = tf.keras.layers.Dense(d_model)
        self.Wq = tf.keras.layers.Dense(d_model)

        self.sdpa = ScaledDotProductAttention(d_model)

        self.linear = tf.keras.layers.Dense(d_model)

    def split_heads(self, X, batch_size):
        # X -> (batch_size, seq_len, d_model)

        # split -> (batch_size, seq_len, num_heads, num_heads/depth)
        split = tf.reshape(X, (batch_size, -1, self.num_heads, self.head_depth))

        # swap -> (batch_size, num_heads, seq_len, num_heads/depth)
        swap = tf.transpose(split, perm=[0, 2, 1, 3])
        return swap

    # Inverse of split_heads function
    def concat_heads(self, X, batch_size):
        # X -> (batch_size, num_heads, seq_len, num_heads/depth)

        # swap -> (batch_size, seq_len, num_heads, num_heads/depth)
        swap = tf.transpose(X, perm=[0, 2, 1, 3])

        # concat -> (batch_size, seq_len, d_model)
        concat = tf.reshape(swap, (batch_size, -1, self.d_model))
        return concat

    # noinspection PyMethodOverriding
    def call(self, K, V, Q, mask):
        batch_size = tf.shape(Q)[0]

        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.Wq(Q)

        K_split = self.split_heads(K, batch_size)
        V_split = self.split_heads(V, batch_size)
        Q_split = self.split_heads(Q, batch_size)

        attention_out = self.sdpa(K_split, V_split, Q_split, mask)

        X = self.concat_heads(attention_out, batch_size)
        X = self.linear(X)

        return X


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, d_ff, dropout_rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    # noinspection PyMethodOverriding
    def call(self, inputs, training, mask):
        X = inputs
        attention_out = self.mha(X, X, X, mask)
        attention_out = self.dropout1(attention_out, training=training)
        X = self.layer_norm1(X + attention_out)

        ffn_out = self.ffn(X)
        ffn_out = self.dropout2(ffn_out, training=training)
        X = self.layer_norm2(X + ffn_out)
        return X


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_pe, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_pe, d_model)

        self.encoder_layers = [EncoderLayer(num_heads, d_model, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # noinspection PyMethodOverriding
    def call(self, inputs, training, padding_mask):
        # inputs -> (batch_size, input_seq_len)
        seq_len = tf.shape(inputs)[1]

        X = self.embedding(inputs)
        X *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        X += positional_encoding[:, :seq_len, :]

        X = self.dropout(X, training=training)

        for i in range(self.num_layers):
            X = self.encoder_layers[i](X, training, padding_mask)

        # X -> (batch_size, input_seq_len, d_model)
        return X


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.masked_mha = MultiHeadAttention(num_heads, d_model)
        self.mha = MultiHeadAttention(num_heads, d_model)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    # noinspection PyMethodOverriding
    def call(self, inputs, encoder_outputs, training, look_ahead_mask, padding_mask):
        # inputs -> (batch_size, pred_len, d_model)
        # encoder_outputs -> (batch_size, input_seq_len, d_model)
        X = inputs
        masked_mha_out = self.masked_mha(X, X, X, look_ahead_mask)
        masked_mha_out = self.dropout1(masked_mha_out, training=training)
        X = self.layer_norm1(X + masked_mha_out)

        mha_out = self.mha(encoder_outputs, encoder_outputs, X, padding_mask)
        mha_out = self.dropout2(mha_out, training=training)
        X = self.layer_norm2(X + mha_out)

        ffn_out = self.ffn(X)
        ffn_out = self.dropout3(ffn_out, training=training)
        X = self.layer_norm3(X + ffn_out)

        # X -> (batch_size, pred_len, d_model)
        return X


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_pe, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_pe, d_model)

        self.decoder_layers = [DecoderLayer(num_heads, d_model, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # noinspection PyMethodOverriding
    def call(self, inputs, encoder_outputs, training, look_ahead_mask, padding_mask):
        # inputs -> (batch_size, pred_len)
        seq_len = tf.shape(inputs)[1]

        X = self.embedding(inputs)
        X *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        X += positional_encoding[:, :seq_len, :]

        X = self.dropout(X, training=training)

        for i in range(self.num_layers):
            X = self.encoder_layers[i](X, encoder_outputs, training, look_ahead_mask, padding_mask)

        # X -> (batch_size, pred_len, d_model)
        return X


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        # TODO: figure out tokenizer
        self.d_model = d_model

        self.tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization()
        vocab_size = len(self.tokenizer.get_vocabulary())

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff,
                               vocab_size, max_pe=vocab_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff,
                               vocab_size, max_pe=vocab_size, dropout_rate=dropout_rate)

        self.linear = tf.keras.layers.Dense(vocab_size)

    # noinspection PyMethodOverriding
    def call(self, inputs, target, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        :param inputs: Previous words ----------------------------> (batch_size, input_seq_len)
        :param target: Next words to predict ---------------------> (batch_size, pred_len)
        :param training: Whether to apply dropout ----------------> (bool)
        :param enc_padding_mask: Mask for encoder ----------------> (batch_size, 1, 1, max_seq_len)
        :param look_ahead_mask: Mask for first MHA in decoder ----> (batch_size, 1, pred_len-1, pred_len-1)
        :param dec_padding_mask: Mask for second MHA in decoder --> (batch_size, 1, 1, max_seq_len)

        :return: Output probabilities for next word --------------> (batch_size, pred_len, vocab_size)
        """
        encoder_out = self.encoder(inputs, training, enc_padding_mask)
        decoder_out = self.decoder(target, encoder_out, training, look_ahead_mask, dec_padding_mask)
        out = self.linear(decoder_out)

        return out


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, loss_fn):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_fn(real, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
test_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar, transformer, optimizer, loss_fn):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions = transformer(inp, tar_inp, True,
                                  enc_padding_mask,
                                  combined_mask,
                                  dec_padding_mask)
        loss = loss_function(tar_real, predictions, loss_fn)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


# @tf.function(input_signature=test_step_signature)
def test_step(inp, tar, transformer, loss_fn):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions = transformer(inp, tar_inp, False,
                              enc_padding_mask,
                              combined_mask,
                              dec_padding_mask)
    loss = loss_function(tar_real, predictions, loss_fn)

    test_loss(loss)
    test_accuracy(accuracy_function(tar_real, predictions))


def train_loop(model, train_dataset, test_dataset, epochs, log_freq, log_dir):
    learning_rate = CustomSchedule(model.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, 'transformer', current_time, 'train')
    test_log_dir = os.path.join(log_dir, 'transformer', current_time, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar, model, optimizer, loss_fn)

            if batch % log_freq == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=batch)
                    tf.summary.scalar('accuracy', train_accuracy.result(), step=batch)
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        for (batch, (inp, tar)) in enumerate(test_dataset):
            test_step(inp, tar, model, loss_fn)

            if batch % log_freq == 0:
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss.result(), step=batch)
                    tf.summary.scalar('accuracy', test_accuracy.result(), step=batch)

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        elapsed_time = str(datetime.timedelta(time.time() - start))
        print('Time taken for 1 epoch: {} secs\n'.format(elapsed_time))
