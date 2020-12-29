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

from google.cloud import storage


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--force-cpu',
        type=bool,
        required=False,
        default=False,
        help='Forces model onto CPU on an NVIDIA GPU, default=False')

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
        '--num-batches',
        default=5000,
        type=int,
        help='number of batches to generate from the dataset: steps = 0.8 * batch_num, default=5000')
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=1000,
        help='maximum number of words in encoder vocabulary, default=1000')
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=5,
        help='number of words to predict from, default=5')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    parser.add_argument(
        '--data-path',
        required=True,
        type=str,
        help='Path to data')
    parser.add_argument(
        '--cloud',
        type=bool,
        default=False,
        help='Run on cloud')
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
    if args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    data_splits, tokenizer = input_fn(buffer_size=args.buffer_size, batch_size=args.batch_size,
                                      data_path=args.data_path,
                                      min_sentence_length=3, num_batches=args.num_batches,
                                      vocabulary_size=args.vocab_size)

    train_dataset = data_splits['train']
    val_dataset = data_splits['val']
    test_dataset = data_splits['test']

    lstm_model = LSTMModel(tokenizer, alpha=1, beta=0)

    # print(tokenizer.get_vocabulary())

    # for x, y in train_dataset.__iter__():
    #     print(x, y)
    #     # print(lstm_model.loss(y, lstm_model(x)))
    #     break

    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=lstm_model.loss,
                       metrics=[tf.keras.metrics.CategoricalAccuracy()])

    tensorboard_callback = None
    cloud = args.cloud
    # cloud = False
    if cloud:
        print('cloud')
        log_dir = os.path.join('gs://speech-predictor-bucket', 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=1000, profile_batch=0)
        tensorboard_callback = [tensorboard_callback]

    lstm_model.fit(train_dataset, epochs=args.num_epochs, validation_data=val_dataset,
                   callbacks=tensorboard_callback)

    export_path = os.path.join(args.job_dir, 'predict_lstm')
    tf.keras.models.save_model(lstm_model, export_path)
    print('Model exported to: {}'.format(export_path))

    # lstm_model = tf.keras.models.load_model(export_path, compile=False)
    #
    # eval_text = ["the "]
    # prediction = lstm_model.predict(eval_text)
    # top_5 = tf.math.top_k(prediction, 5)
    # for t in top_5[1][0]:
    #     index = tf.get_static_value(t)
    #     print(eval_text[0] + ' ... ' + lstm_model.tokenizer.get_vocabulary()[index])
    # # print(top_5[1][0])
    # pred_idx = np.argmax(prediction[:, 2:])
    # prediction = lstm_model.tokenizer.get_vocabulary()[pred_idx+2]
    # x = lstm_model.tokenizer(eval_text)
    # # print(x)
    # x = lstm_model.embedding(x)
    # # print(x)
    # x = lstm_model.lstm(x)
    # # print(x)
    # print(eval_text[0] + ' ... ' + prediction)


if __name__ == '__main__':
    args = get_args()
    print(args)
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
