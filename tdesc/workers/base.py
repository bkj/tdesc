#!/usr/bin/env python

"""
    base.py
    
    !! Need to figure out how to handle timeouts, since sometimes this wants
    to be run on a stream coming from the internet that's slow or bursty
"""

import os
import sys
import json
import itertools
from time import time
from concurrent.futures import ThreadPoolExecutor

class BaseWorker(object):
    
    print_interval = 25
    
    def run(self, io_threads, timeout, chunk_size=10000):
        start_time = time()
        pool = ThreadPoolExecutor(max_workers=io_threads)
        gen = (line.strip() for line in sys.stdin)
        i = 0
        for chunk in self._chunker(gen, chunk_size):
            for obj in pool.map(self.do_io, chunk):
                i += 1
                if obj[1] is not None:
                    yield self.featurize(*obj)
                    
                    if not i % self.print_interval:
                        print >> sys.stderr, json.dumps({
                            "i" : i,
                            "time" : time() - start_time,
                        })
            
        self.close()
        
    def _chunker(self, iterable, chunk_size):
        while True:
            yield itertools.chain([iterable.next()], itertools.islice(iterable, chunk_size-1))
    
    def do_io(self, req):
        try:
            return (req, self.imread(req))
        except:
            return (req, None)
    
    def close(self):
        pass