import numpy as np
import tensorflow as tf

from trainer import model as m
import os


if __name__ == '__main__':
    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    train_dataset = m.input_fn(64)

    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = m.create_keras_model(encoder)

    history = model.fit(train_dataset, epochs=1)
    model.summary()

    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

