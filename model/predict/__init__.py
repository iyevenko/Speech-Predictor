import tensorflow as tf
from google.cloud import storage
import os

# BUCKET_ADDRESS = 'gs://speech-predictor-bucket/keras-job-dir/universal-sentence-encoder/next_word_predictor'
# BUCKET_ADDRESS = 'gs://speech-predictor-bucket/keras-job-dir/predict-lstm/v2'

client = storage.Client()
bucket = client.bucket('speech-predictor-bucket')
blob = bucket.blob('keras-job-dir/universal-sentence-encoder/vocab/vocab-10000.txt')
vocab = blob.download_as_text().split('\n')[:-1]

print('Downloading Model')
blobs = bucket.list_blobs(prefix='keras-job-dir/universal-sentence-encoder/next_word_predictor')
folder = './saved_models'

if not os.path.exists(folder):
        os.mkdir(folder)
if not os.path.exists(folder + '/variables'):
        os.mkdir(folder+'/variables')
for blob in blobs:
        path = 'keras-job-dir/universal-sentence-encoder/next_word_predictor/'

        file_name = os.path.join(folder, blob.name[len(path):])

        blob.download_to_filename(file_name)

print('Loading Model')
model = tf.keras.models.load_model(folder, compile=False)
print('Loading Model')
#
# print('Loading Model.')
# model = tf.keras.models.load_model(BUCKET_ADDRESS, compile=False)
# print('Model Loaded.')
