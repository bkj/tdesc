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

class BaseWorker(object):
    
    print_interval = 25
    
    def run(self, io_threads, timeout):
        self.timeout = timeout
        self.filename_queue = Queue()
        self.io_queue = Queue()
        
        # Thread to read stdin
        newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
        stdin_reader = Process(target=read_stdin, args=(newstdin, self.filename_queue))
        stdin_reader.start()
        
        # Thread(s) to load images    
        image_processors = [Process(target=self.do_io) for _ in range(io_threads)]
        for image_processor in image_processors:
            image_processor.start()
        
        # "Thread" to do work
        for w in self.do_work():
            yield w
    
    def do_io(self):
        while True:
            try:
                req = self.filename_queue.get(timeout=self.timeout)
                try:
                    img = self.imread(req)
                    if img is not None:
                        self.io_queue.put((req, img))
                except KeyboardInterrupt:
                    raise
                except:
                    print >> sys.stderr, "do_io: Error at %s" % req
            except KeyboardInterrupt:
                raise
                os._exit(0)
            except Empty:
                return
    
    def do_work(self):
        i = 0
        start_time = time()
        while True:
            try:
                req, obj = self.io_queue.get(timeout=self.timeout)
                yield self.featurize(req, obj)
                
                i += 1
                if not i % self.print_interval:
                    print >> sys.stderr, "%d images | %f seconds " % (i, time() - start_time)
            except KeyboardInterrupt:
                raise
                os._exit(0)
            except Empty:
                self.close()
                os._exit(0)
            except Exception as e:
                raise e
                os._exit(0)
    
    def close(self):
        pass