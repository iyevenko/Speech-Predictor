import tensorflow_datasets as tfds
import tensorflow as tf
import os
import numpy as np

model = tf.keras.models.load_model(os.path.join('saved_models', 'lstm_3'))
model.summary()

sample_text = ('The movie was really awful. A child could have done the animation and the graphics.'
                 'I would not recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions)
