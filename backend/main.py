# [START gae_python38_app]
# [START gae_python3_app]
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import requests

# If `entrypoint` is not defined in dispatch.yaml, App Engine will look for an app
# called `app` in `main.py`.
import predict

app = Flask(__name__)
CORS(app)

@app.route('/rest/hello')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.route('/rest/process-text/<text>')
def process_text(text):
    """Return a friendly HTTP greeting."""
    response = text.upper()
    print(response)
    return {
        'response': response
    }

@app.route('/rest/next-word/')
def next_word():
    text = request.headers['text']

    response = (requests.get('https://predict-sy46lv4e6q-ue.a.run.app/predict', headers={"text": text}).content).decode('utf-8')
    return {
        'response': response
    }
    # response.headers.add('Access-Control-Allow-Origin', '*')

# @app.route('/rest/review-sentiment/')
# def review_sentiment():
#     """Return a friendly HTTP greeting."""
#     response, value = predict.get_sentiment(request.headers['text'])
#     print(response)
#     return {
#         'response': response,
#         'value': value
#     }

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to dispatch.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)

# @app.after_request
# def after_request(response):
#   response.headers.add('Access-Control-Allow-Origin', '*')
#   response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#   response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#   return response

# [END gae_python3_app]
# [END gae_python38_app]
