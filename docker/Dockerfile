FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y    apt-utils \
                                    lsb-release \
                                    mesa-utils \
                                    gnupg2 \
                                    net-tools \
                                    iputils-ping \
                                    build-essential \
                                    wget \
                                    unzip \
                                    curl \
                                    git \
                                    mc \
                                    vim \
                                    nano \
                                    ffmpeg \
                                    x11-apps

## GUI tunnel from docker
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0"


RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 \
        libglx0 libglx0:i386 \
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*
# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# RUN apt update && apt install -y --allow-change-held-packages libcudnn8

RUN apt update &&  apt install -y make \
                                libssl-dev \
                                zlib1g-dev \
                                libbz2-dev \
                                libreadline-dev \
                                libsqlite3-dev \
                                llvm \
                                libncurses5-dev \
                                libncursesw5-dev \
                                xz-utils \
                                tk-dev \
                                libffi-dev \
                                liblzma-dev \
                                libgdbm-dev \
                                libnss3-dev \
                                libedit-dev \
                                libc6-dev \
                                cmake \
                                unzip \
                                autoconf \
                                autogen \
                                automake \
                                libtool \
                                mlocate

RUN apt update && apt install -y --allow-downgrades libcudnn8=8.8.1.3-1+cuda11.8 libcudnn8-dev=8.8.1.3-1+cuda11.8

RUN apt update && apt install -y python3-pip python3-dev && pip3 install --upgrade setuptools pip
RUN apt update && apt install -y python3-tk
ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir gdown --pre
RUN pip3 install --no-cache-dir ftfy \
                regex \
                fvcore \
                moviepy \
                flax==0.5.1 \
                openai \
                easydict \
                jax==0.4.2 \
                jaxlib==0.4.2 \
                pycocotools \
                tflite-model-maker

RUN pip3 install --no-cache-dir tensorflow==2.12.0 tensorboard
RUN pip3 install --no-cache-dir opencv-contrib-python
WORKDIR /
RUN pip3 install imageio==2.4.1 imageio-ffmpeg
RUN wget https://github.com/imageio/imageio-binaries/raw/master/ffmpeg/ffmpeg-linux64-v3.3.1 && \
    mkdir /root/.imageio && \
    mv /ffmpeg-linux64-v3.3.1 /root/.imageio/ffmpeg
RUN pip3 install --no-cache-dir torch==2.0.1 torchvision==0.15.2 

RUN pip3 install --no-cache-dir transformers \
                            addict \
                            yapf \
                            timm \
                            supervision==0.6.0 \
                            accelerate \
                            scipy \
                            safetensors \
                            segments-ai \
                            scikit-image 
                            
RUN pip3 install --no-cache-dir onnxruntime onnx
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git
WORKDIR /
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
ENV TORCH_CUDA_ARCH_LIST "6.0 6.1 7.0 7.5 8.0"
ENV BUILD_WITH_CUDA True
ENV CUDA_HOME /usr/local/cuda-11.7/ 
RUN pip3 install /GroundingDINO
RUN pip install grpcio grpcio-tools fastapi requests
ENV TOKENIZER_PARALLELISM false
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-c"]