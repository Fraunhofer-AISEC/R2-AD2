# Check https://www.tensorflow.org/install/source#gpu
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip

# Install python packages
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
