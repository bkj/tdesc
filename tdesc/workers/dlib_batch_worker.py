#!/usr/bin/env python

"""
    workers.py
"""

from __future__ import print_function

import sys
import numpy as np
import dlib
from dlib_worker import DlibFaceWorker


class DlibFaceBatchWorker(DlibFaceWorker):
    """
        compute dlib face descriptors
    """

    def __init__(self, num_jitters=10, dnn=False, det_threshold=0.0, upsample=0, batch_size=10):
        super(DlibFaceBatchWorker, self).__init__(
            num_jitters=num_jitters, 
            dnn=dnn,
            det_threshold=det_threshold,
            upsample=upsample
        )
        self.batch_size = batch_size
        
        
        self.path_buffer = []
        self.img_buffer = []
        self.det_buffer = []
        
        print('DlibFaceBatchWorker: ready (dnn=%d | num_jitters=%d)' % (1, int(num_jitters)), file=sys.stderr)
        
    def featurize(self, path, obj, return_feat=False):
        img, dets = obj
        
        self.path_buffer.append(path)
        self.img_buffer.append(img)
        self.det_buffer.append(dets)
        
        if not len(self.img_buffer) % self.batch_size:
            self._featurize()
            
    def _featurize(self):
        # all_dets, _ = zip(*self.detector(self.img_buffer))
        
        all_shapes = []
        for img, dets in zip(self.img_buffer, self.det_buffer):
            all_shapes.append([self.sp(img, det) for det in dets])
            
        all_face_descriptors = self.facerec.compute_batch_face_descriptors(self.img_buffer, all_shapes)
        
        i = 0
        for path, dets in zip(self.path_buffer, self.det_buffer):
            for ind, det in enumerate(dets):
                face_descriptor = all_face_descriptors[i]
                print('\t'.join((
                    path,
                    str(ind),
                    '\t'.join(map(str, [det.top(), det.bottom(), det.left(), det.right()])),
                    '\t'.join(map(str, face_descriptor))
                )))
                sys.stdout.flush()
                i += 1
                
        self.path_buffer = []
        self.img_buffer = []
        self.det_buffer = []
        
    def close(self):
        if len(self.path_buffer) > 0:
            self._featurize()
        print('DlibFaceBatchWorker: terminating', file=sys.stderr)
