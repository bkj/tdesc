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
    
    def __init__(self, num_jitters=10):
        super(DlibFaceBatchWorker, self).__init__(num_jitters=num_jitters, dnn=True)
        self.num_jitters = num_jitters
        
        self.path_buffer = []
        self.img_buffer = []
        
        print >> sys.stderr, 'DlibFaceBatchWorker: ready (dnn=%d | num_jitters=%d)' % (1, int(num_jitters))
    
    def featurize(self, path, obj, return_feat=False):
        img, _ = obj
        
        self.path_buffer.append(path)
        self.img_buffer.append(img)
        
        if not len(self.img_buffer) % 10:
            all_dets, _ = zip(*self.detector(self.img_buffer))
            all_inds = np.hstack(map(lambda x: range(len(x)), all_dets))
            
            all_shapes = []
            for img, dets in zip(self.img_buffer, all_dets):
                all_shapes.append([self.sp(img, det) for det in dets])
            
            all_face_descriptors = self.facerec.compute_batch_face_descriptors(self.img_buffer, all_shapes)
            all_dets = np.hstack(all_dets)
            
            iter_ = zip(self.path_buffer, all_inds, all_dets, all_face_descriptors)
            for path, ind, det, face_descriptor in iter_:
                print '\t'.join((
                    path, 
                    str(ind),
                    '\t'.join(map(str, [det.top(), det.bottom(), det.left(), det.right()])), 
                    '\t'.join(map(str, face_descriptor))
                ))
                sys.stdout.flush()
            
            self.path_buffer = []
            self.img_buffer = []

    
    def close(self):
        print >> sys.stderr, 'DlibFaceBatchWorker: terminating'
