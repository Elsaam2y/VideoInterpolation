FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get install -y python3-pip && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

COPY . /app
WORKDIR /app