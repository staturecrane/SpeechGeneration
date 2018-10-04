FROM python:3.6.1-slim
MAINTAINER FunnelAI <rich@funnelai.co>

RUN apt-get update && apt-get install -y \
    bzip2 \
    curl \
    ca-certificates \
    git \
    libx11-6 \
    sudo \
    wget \
    unzip

WORKDIR /app
ENV PYTHONPATH /app 
ENV FLASK_APP main.py

RUN pip3 install -q gensim
COPY requirements.txt /app
RUN pip3 install -q -r requirements.txt

COPY main.py /app
COPY librispeech* /app/

EXPOSE 8000

CMD flask run -h 0.0.0.0 -p 8000
