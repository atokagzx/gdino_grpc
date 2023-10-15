#!/bin/bash

xhost +local:docker || true

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
docker run -ti --rm \
      --gpus all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e "DISPLAY" \
      -e "QT_X11_NO_MITSHM=1" \
      -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      -e XAUTHORITY \
      -v $ROOT_DIR:/workspace \
      -v $HOME/.cache/gdino_grpc:/root/.cache \
      -v /dev:/dev \
      --net=host \
      --privileged \
      --name panoptic_exp panoptic_exp-img \
      bash