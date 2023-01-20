FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        git \
        wget \
        g++ \
        gcc \
        ca-certificates \
        openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Compile python from source - avoid unsupported library problems
RUN apt-get update && apt-get install -y \
        build-essential checkinstall \
        libreadline-dev \
        libncursesw5-dev \
        libssl-dev \
        libsqlite3-dev \
        tk-dev \
        libgdbm-dev \
        libc6-dev \
        libbz2-dev \
        libffi-dev \
        zlib1g-dev \
    && cd /usr/src \
    && wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz \
    && tar xzf Python-3.8.10.tgz \
    && cd Python-3.8.10 \
    && ./configure \
    && make

RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

CMD ["/bin/bash"]