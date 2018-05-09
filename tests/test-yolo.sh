#!/bin/bash

# test-yolo.sh
#
# Test tdesc.yolo

# Need to install darknet first:
# 
# git clone https://github.com/bkj/darknet
# cd darknet
# ./install.sh


export PYTHONPATH="path/to/darknet/.so/files"
# export PYTHONPATH="~projects/darknet/pyDarknet/"

YOLO_CFG_PATH="~/.tdesc/yolo/yolo-custom.cfg"
YOLO_WEIGHT_PATH="~/.tdesc/yolo/yolo-custom_final.weights"
YOLO_NAME_PATH="~/.tdesc/yolo/custom.names"

IMG_PATH="*/images"
find $IMG_PATH -type f |\
    python -m tdesc --model yolo \
        --yolo-cfg-path $YOLO_CFG_PATH \
        --yolo-weight-path $YOLO_WEIGHT_PATH \
        --yolo-name-path $YOLO_NAME_PATH > res
