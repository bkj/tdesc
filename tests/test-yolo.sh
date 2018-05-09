#!/bin/bash

# test-yolo.sh
#
# Test tdesc.yolo
# !! This might be tricky to install -- open an issue for support.

export PYTHONPATH="path/to/darknet/.so/files"

YOLO_CFG_PATH="~/.tdesc/yolo/yolo-custom.cfg"
YOLO_WEIGHT_PATH="~/.tdesc/yolo/yolo-custom_final.weights"
YOLO_NAME_PATH="~/.tdesc/yolo/custom.names"

IMG_PATH="/srv/beta/projects/darknet-bkj/custom-tools/pfr-data/scratch/*/images/"
find $IMG_PATH -type f |\
    python -m tdesc --model yolo \
        --yolo-cfg-path $YOLO_CFG_PATH \
        --yolo-weight-path $YOLO_WEIGHT_PATH \
        --yolo-name-path $YOLO_NAME_PATH > res
