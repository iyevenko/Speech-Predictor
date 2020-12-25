from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

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

    train_dataset, tokenizer = input_fn(buffer_size=args.buffer_size, batch_size=args.batch_size,
                                        data_path=os.path.join('..', '..', 'ANC_training_data'),
                                        min_sentence_length=10, dataset_fraction=0.1, vocabulary_size=args.vocab_size)

    lstm_model = LSTMModel(tokenizer, alpha=0.1, beta=0.5)

    # for x, y in train_dataset.__iter__():
    #     print(lstm_model.loss(y, lstm_model(x)))
    #     break

    #
    lstm_model.compile(optimizer='adam', loss=lstm_model.loss)

    lstm_model.fit(train_dataset, epochs=args.num_epochs)

    export_path = os.path.join(args.job_dir, 'predict_lstm')
    tf.keras.models.save_model(lstm_model, export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
