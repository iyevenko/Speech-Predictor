import os

import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)

@app.route("/predict")
def predict_word():
    text = request.headers['text']
    prediction = model.predict([text])
    top_k = tf.math.top_k(prediction, 5)[1][0]
    top_words = [model.tokenizer.get_vocabulary()[tf.get_static_value(idx)] for idx in top_k]
    print(top_words)
    return {
        'response': top_words
    }

if __name__ == "__main__":
    BUCKET_ADDRESS = 'gs://speech-predictor-bucket/keras-job-dir/predict_lstm/'
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    print('PREDICT_APP: Loading Model', flush=True)
    model = tf.keras.models.load_model(BUCKET_ADDRESS, compile=False)
    print('PREDICT_APP: Model Loaded', flush=True)
