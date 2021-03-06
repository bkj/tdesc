#!/usr/bin/env python

"""
    yolo_worker.py
"""

from __future__ import print_function

import os
import sys
import urllib
import contextlib
import cStringIO
import numpy as np
from PIL import Image

from base import BaseWorker


def import_yolo():
    global DarknetObjectDetector
    from libpydarknet import DarknetObjectDetector


class DetBBox(object):

    def __init__(self, bbox):
        self.left = bbox.left
        self.right = bbox.right
        self.top = bbox.top
        self.bottom = bbox.bottom
        self.confidence = bbox.confidence
        self.cls = bbox.cls


class YoloWorker(BaseWorker):
    def __init__(self, cfg_path, weight_path, name_path, thresh=0.1, nms=0.3, target_dim=416):
        import_yolo()
        
        cfg_path    = os.path.expanduser(cfg_path)
        weight_path = os.path.expanduser(weight_path)
        name_path   = os.path.expanduser(name_path)
        
        DarknetObjectDetector.set_device(0)
        self.target_dim = target_dim
        self.class_names = open(name_path).read().splitlines()
        self.det = DarknetObjectDetector(cfg_path, weight_path, thresh, nms, 0)
        print('YoloWorker: ready', file=sys.stderr)

    def imread(self, path):
        if path[:4] == 'http':
            with contextlib.closing(urllib.urlopen(path)) as req:
                path = cStringIO.StringIO(req.read())
        img = Image.open(path).convert('RGB')
        img = img.resize((self.target_dim, self.target_dim), Image.BILINEAR)

        data = np.array(img).transpose([2,0,1]).astype(np.uint8).tostring()
        return data, (img.size[0], img.size[1])

    def featurize(self, meta, obj, return_feat=False):
        data, size = obj
        bboxes = [DetBBox(x) for x in self.det.detect_object(data, size[0], size[1], 3).content]
        feats = []
        for bbox in bboxes:
            class_name = self.class_names[bbox.cls]
            if not return_feat:
                print('\t'.join(map(str, [
                    meta,
                    class_name,
                    bbox.confidence,
                    bbox.top,
                    bbox.bottom,
                    bbox.left,
                    bbox.right
                ])))
                sys.stdout.flush()
            else:
                feats.append({
                    "class_name" : class_name,
                    "confidence" : bbox.confidence,
                    "bbox" : [bbox.top, bbox.bottom, bbox.left, bbox.right],
                })

        if return_feat:
            return meta, feats

    def close(self):
        print('YoloWorker: terminating', file=sys.stderr)
        pass
