#!/usr/bin/env python

"""
    base.py
    
    !! Need to figure out how to handle timeouts, since sometimes this wants
    to be run on a stream coming from the internet that's slow or bursty
"""

import os
import sys
from time import time
from concurrent.futures import ThreadPoolExecutor

class BaseWorker(object):
    
    print_interval = 25
    
    def run(self, io_threads, timeout):
        start_time = time()
        pool = ThreadPoolExecutor(max_workers=io_threads)
        sys.stdin = (line.strip() for line in sys.stdin)
        for i, obj in enumerate(pool.map(self.do_io, sys.stdin)):
            if obj[1] is not None:
                yield self.featurize(*obj)
                
                if not i % self.print_interval:
                    print >> sys.stderr, "%d images | %f seconds " % (i, time() - start_time)
        
        self.close()
    
    def do_io(self, req):
        return (req, self.imread(req))
    
    def close(self):
        pass