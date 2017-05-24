#!/usr/bin/env python

"""
    base.py
"""

import os
import sys
import urllib
import cStringIO
from time import time
from Queue import Empty
from multiprocessing import Process, Queue

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


def do_work(worker, io_queue, timeout, print_interval=25):
    i = 0
    start_time = time()
    while True:
        try:
            path, obj = processed_images.get(timeout=timeout)
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


class BaseWorker(object):
    
    def run(self, io_threads=1, timeout=10):
        # Thread to read filenames from stdin
        filenames = Queue()
        newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
        stdin_reader = Process(target=read_stdin, args=(newstdin, filenames))
        stdin_reader.start()
        
        # Thread to load images    
        processed_images = Queue()
        image_processors = [Process(target=do_io, args=(filenames, processed_images, self.worker.imread, timeout)) for _ in range(io_threads)]
        for image_processor in image_processors:
            image_processor.start()
        
        # "Thread" to do work
        do_work(worker=self.worker, io_queue=processed_images, timeout=timeout)