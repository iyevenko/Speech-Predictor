#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-4
#FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
FROM python:3.8-slim
#FROM tensorflow/serving

ENV PYTHONUNBUFFERED True

ENV APP_HOME /predict
WORKDIR $APP_HOME
COPY . ./

RUN pip install Flask flask_cors gunicorn tensorflow google-cloud-storage

RUN mv setup.py ..
WORKDIR ..
RUN python3 setup.py build
RUN python3 setup.py install
RUN pip install predict
WORKDIR $APP_HOME

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

#COPY ../requirements.txt /home/requirements.txt
#
#RUN pip install -r /home/requirements.txt

#VOLUME ../data ../data
#
#ENTRYPOINT task.py --job-dir ../.. --num-epochs 10 --batch-size 128 --vocab-size 30000 --buffer-size 5
