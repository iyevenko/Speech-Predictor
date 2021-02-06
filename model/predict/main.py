import os

import tensorflow as tf
from flask import Flask, request
from flask_cors import CORS

# from predict import model
from predict import model, vocab

app = Flask(__name__)
CORS(app)

# @app.route("/predict")
def predict_word():
    text = request.headers['text']
    prediction = model.predict([text])
    top_k = tf.math.top_k(prediction, 5)[1][0]
    top_words = [model.tokenizer.get_vocabulary()[tf.get_static_value(idx)] for idx in top_k]
    print(top_words)
    return {
        'response': top_words
    }

@app.route("/predict")
def predict_word_USE():
    prev = request.headers['text']
    logits = model((tf.constant([[prev]]), tf.constant([[prev.split(' ')[-1]]])), training=False)
    top_k = tf.math.top_k(logits, 5)[1][0]
    top_words = [vocab[top_k[i]] for i in range(top_k.shape[0])]
    top_words.remove('<UNK>')

    return {
        'response': top_words
    }



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
