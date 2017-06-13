#!/usr/bin/env python

"""
    workers.py
"""

import sys
import numpy as np

from dlib_worker import DlibFaceWorker

class DlibFaceBatchWorker(DlibFaceWorker):
    """ 
        compute dlib face descriptors 
    """
    
    def __init__(self, num_jitters=10, batch_size=10):
        super(DlibFaceBatchWorker, self).__init__(num_jitters=num_jitters, dnn=True)
        self.num_jitters = num_jitters
        self.batch_size = batch_size
        
        self.path_buffer = []
        self.img_buffer = []
        
        print >> sys.stderr, 'DlibFaceBatchWorker: ready (dnn=%d | num_jitters=%d)' % (1, int(num_jitters))
    
    def featurize(self, path, obj, return_feat=False):
        img, _ = obj
        
        self.path_buffer.append(path)
        self.img_buffer.append(img)
        
        if not len(self.img_buffer) % self.batch_size:
            self._featurize()
    
    def _featurize(self):
        all_dets, _ = zip(*self.detector(self.img_buffer))
        
        all_shapes = []
        for img, dets in zip(self.img_buffer, all_dets):
            all_shapes.append([self.sp(img, det) for det in dets])
        
        all_face_descriptors = self.facerec.compute_batch_face_descriptors(self.img_buffer, all_shapes)
        
        i = 0
        for path, dets in zip(self.path_buffer, all_dets):
            for ind, det in enumerate(dets):
                face_descriptor = all_face_descriptors[i]
                print '\t'.join((
                    path, 
                    str(ind),
                    '\t'.join(map(str, [det.top(), det.bottom(), det.left(), det.right()])), 
                    '\t'.join(map(str, face_descriptor))
                ))
                sys.stdout.flush()
                i += 1
        
        self.path_buffer = []
        self.img_buffer = []
    
    def close(self):
        if len(self.path_buffer) > 0:
            self._featurize()
        print >> sys.stderr, 'DlibFaceBatchWorker: terminating'
