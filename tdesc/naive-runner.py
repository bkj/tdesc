#!/usr/bin/env python

"""
    naive-runny.py
"""

import os
import sys
import argparse
import numpy as np

import urllib
import cStringIO
from time import time

from workers import VGG16Worker, DlibFaceWorker

# --
# Init

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg16')
    # VGG16 options
    parser.add_argument('--crow', action="store_true")
    # DlibFace options
    parser.add_argument('--outpath', type=str, default='./db.h5')
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    if args.model == 'vgg16':
        worker = VGG16Worker(args.crow)
    elif args.model == 'dlib_face':
        worker = DlibFaceWorker(args.outpath)
    else:
        raise Exception()
    
    start_time = time()
    for i, line in enumerate(sys.stdin):
        path = line.strip()
        
        try:
            worker.featurize(path, worker.imread(path))
            
            if not (i + 1) % 100:
                print >> sys.stderr, "%d images | %f seconds " % (i, time() - start_time)
            
        except KeyboardInterrupt:
            raise
            os._exit(0)
        except Exception as e:
            raise e
            os._exit(0)
    
    print >> sys.stderr, "%d images | %f seconds " % (i, time() - start_time)
