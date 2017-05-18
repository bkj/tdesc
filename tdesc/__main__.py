#!/usr/bin/env python

"""
    runner.py
"""

import os
import sys
import argparse
import numpy as np

import urllib
import cStringIO
from time import time

from multiprocessing import Process, Queue
from Queue import Empty

# --
# Init

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--io-threads', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=10)
    
    # VGG16 options
    parser.add_argument('--crow', action="store_true")
    
    # DlibFace options
    parser.add_argument('--outpath', type=str, default='./db.h5')
    
    # Yolo options
    parser.add_argument('--yolo-cfg-path', type=str, required=False)
    parser.add_argument('--yolo-weight-path', type=str, required=False)
    parser.add_argument('--yolo-name-path', type=str, required=False)
    parser.add_argument('--yolo-thresh', type=float, default=0.1)
    parser.add_argument('--yolo-nms', type=float, default=0.3)
    
    args = parser.parse_args()
    
    if args.model == 'yolo':
        assert args.yolo_cfg_path is not None
        assert args.yolo_weight_path is not None
        assert args.yolo_nms is not None
    
    return args

# --
# Threaded IO

def read_stdin(gen, out_):
    for line in gen:
        out_.put(line.strip())

def do_io(in_, out_, imread, timeout):
    while True:
        try:
            path = in_.get(timeout=timeout)
            try:
                img = imread(path)
                out_.put((path, img))
            except KeyboardInterrupt:
                raise
            except:
                print >> sys.stderr, "do_io: Error at %s" % path
        
        except KeyboardInterrupt:
            raise
            os._exit(0)
        except Empty:
            return

def do_work(worker, io_queue, print_interval=25):
    i = 0
    start_time = time()
    while True:
        
        try:
            path, obj = processed_images.get(timeout=args.timeout)
            worker.featurize(path, obj)
            
            i += 1
            if not i % print_interval:
                print >> sys.stderr, "%d images | %f seconds " % (i, time() - start_time)
            
        except KeyboardInterrupt:
            raise
            os._exit(0)
        except Empty:
            worker.close()
            os._exit(0)
        except Exception as e:
            raise e
            os._exit(0)

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    if args.model == 'vgg16':
        from tdesc.workers import VGG16Worker
        worker = VGG16Worker(args.crow)
    elif args.model == 'dlib_face':
        from tdesc.workers import DlibFaceWorker
        worker = DlibFaceWorker(args.outpath)
    elif args.model == 'yolo':
        from tdesc.workers import YoloWorker
        worker = YoloWorker(**{
            "cfg_path" : args.yolo_cfg_path,
            "weight_path" : args.yolo_weight_path,
            "name_path" : args.yolo_name_path,
            "thresh" : args.yolo_thresh,
            "nms" : args.yolo_nms,
        })
    else:
        raise Exception()
    
    # Thread to read filenames from stdin
    filenames = Queue()
    newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
    stdin_reader = Process(target=read_stdin, args=(newstdin, filenames))
    stdin_reader.start()
    
    # Thread to load images    
    processed_images = Queue()
    image_processors = [Process(target=do_io, args=(filenames, processed_images, worker.imread, args.timeout)) for _ in range(args.io_threads)]
    for image_processor in image_processors:
        image_processor.start()
    
    # "Thread" to do work
    do_work(worker, processed_images)