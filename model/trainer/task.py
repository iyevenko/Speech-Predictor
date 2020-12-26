from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os

import tensorflow as tf
import numpy as np

from trainer.dataset import input_fn
from trainer.model import LSTMModel


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='number of times to go through the data, default=10')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=1000,
        help='maximum number of words in encoder vocabulary, default=1000')
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=5,
        help='maximum number of words in encoder vocabulary, default=1000')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
      args: dictionary of arguments - see get_args() for details
    """

    data_splits, tokenizer = input_fn(buffer_size=args.buffer_size, batch_size=args.batch_size,
                                        data_path=os.path.join('..', '..', 'ANC_training_data'),
                                        min_sentence_length=10, dataset_fraction=0.1, vocabulary_size=args.vocab_size)

    train_dataset = data_splits['train']
    val_dataset = data_splits['val']
    test_dataset = data_splits['test']

    lstm_model = LSTMModel(tokenizer, alpha=0.02, beta=1)

    # for x, y in train_dataset.__iter__():
    #     print(lstm_model.loss(y, lstm_model(x)))
    #     break

    lstm_model.compile(optimizer='adam', loss=lstm_model.loss)

    log_dir = os.path.join('..', 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=500, embeddings_freq=0)

    lstm_model.fit(train_dataset, epochs=args.num_epochs, validation_data=val_dataset, callbacks=[tensorboard_callback])

    export_path = os.path.join(args.job_dir, 'predict_lstm')
    tf.keras.models.save_model(lstm_model, export_path)
    print('Model exported to: {}'.format(export_path))

    # lstm_model = tf.keras.models.load_model(export_path, compile=False)
    #
    # eval_text = ["we like to"]
    # prediction = lstm_model.predict(eval_text)
    # pred_idx = np.argmax(prediction[:, 2:])
    # prediction = lstm_model.tokenizer.get_vocabulary()[pred_idx+2]
    # x = lstm_model.tokenizer(eval_text)
    # print(x)
    # x = lstm_model.embedding(x)
    # print(x)
    # x = lstm_model.lstm(x)
    # print(x)
    # print(eval_text[0] + ' ... ' + prediction)


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
