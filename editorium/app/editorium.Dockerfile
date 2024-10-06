FROM python:3.10

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip

WORKDIR /app/editorium

# RUN pip install -r requirements_versions.txt

ARG USER=editorium
ARG UID=1000
ARG GID=1000
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH

ADD ./requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

ADD ./server /app/editorium/server
ADD ./run-server.sh /app/editorium/run-server.sh

RUN groupadd -g $GID $USER && useradd -m -u $UID -g $GID -s /bin/bash -d /home/$USER $USER  && chown -R $USER:$USER /app/editorium

USER $USER

CMD ["/app/editorium/run-server.sh"]

