#!/bin/bash

# test-yolo.sh
#
# Test tdesc.yolo
# !! This depends on lots of funny business w/ paths, custom builds, etc.  Open an issue for support.

python setup.py clean --all install
export PYTHONPATH=/home/bjohnson/projects/darknet-bkj/pyDarknet/

YOLO_CFG_PATH=/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data.bak/yolo-custom.cfg
YOLO_WEIGHT_PATH=/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data.bak/backup/yolo-custom_final.weights
YOLO_NAME_PATH=/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data.bak/custom.names

find ~/projects/sm-krem/imgs/media/pbs.twimg.com/ -type f | head -n 100 | python -m tdesc --model yolo \
    --yolo-cfg-path $YOLO_CFG_PATH \
    --yolo-weight-path $YOLO_WEIGHT_PATH \
    --yolo-name-path $YOLO_NAME_PATH
