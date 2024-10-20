# FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:22.12-py3 
# disable interactive functions
ENV DEBIAN_FRONTEND=noninteractive
# define timezone utc
ENV TZ=UTC

# install locales
RUN apt-get update && apt-get install -y locales
# generate locales
RUN locale-gen en_US.UTF-8
# set locale
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# FROM python:3.10

RUN apt-get update && apt-get install -y git \
    ffmpeg libsm6 libxext6 python3-dev \
    python3-pip libgl1-mesa-glx libglib2.0-0 \
    libxrender1 libxext6 libsm6 libglib2.0-0 \
    libxext6 libsm6 libxrender-dev \
    build-essential libssl-dev libffi-dev \
    ca-certificates curl wget \
    nano vim htop tmux 

RUN pip install --upgrade pip wheel packaging

WORKDIR /app/editorium

# RUN pip install -r requirements_versions.txt

ARG USER=editorium
ARG UID=1000
ARG GID=1000
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH

# REQUIREMENTS CHANGE NUMBER: 0002
ADD ./requirements1.txt /tmp/requirements1.txt
RUN pip install -r /tmp/requirements1.txt && rm /tmp/requirements1.txt

ADD ./requirements2.txt /tmp/requirements2.txt
RUN pip install -r /tmp/requirements2.txt && rm /tmp/requirements2.txt

ADD ./requirements3.txt /tmp/requirements3.txt
RUN pip install -r /tmp/requirements3.txt && rm /tmp/requirements3.txt

# ADD ./requirements-experimental.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt --force-reinstall
# RUN rm /tmp/requirements.txt

ADD ./server /app/editorium/server
ADD ./run-server.sh /app/editorium/run-server.sh

RUN groupadd -g $GID $USER && useradd -m -u $UID -g $GID -s /bin/bash -d /home/$USER $USER  && chown -R $USER:$USER /app/editorium

USER $USER

ENV TOKENIZERS_PARALLELISM=false
# ENV DS_ACCELERATOR=cpu

CMD ["/app/editorium/run-server.sh"]

