# [START gae_python38_app]
# [START gae_python3_app]
from flask import Flask
from flask_cors import CORS

# If `entrypoint` is not defined in dispatch.yaml, App Engine will look for an app
# called `app` in `main.py`.
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


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to dispatch.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]
