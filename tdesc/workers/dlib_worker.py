#!/usr/bin/env python

"""
    workers.py
"""

import os
import sys
import urllib
import cStringIO
import numpy as np

def import_dlib():
    global dlib
    global h5py
    global io
    import dlib
    import h5py
    from skimage import io

class DlibFaceWorker(object):
    """ 
        compute dlib face descriptors 
    """
    
    def __init__(self, outpath, use_h5py=False):
        import_dlib()
        self.detector = dlib.get_frontal_face_detector()
        
        ppath = os.path.join(os.environ['HOME'], '.tdesc')
        
        shapepath = os.path.join(ppath, 'models/dlib/shape_predictor_68_face_landmarks.dat')
        self.sp = dlib.shape_predictor(shapepath)
        
        facepath = os.path.join(ppath, 'models/dlib/dlib_face_recognition_resnet_model_v1.dat')
        self.facerec = dlib.face_recognition_model_v1(facepath)
        
        if use_h5py:
            self.db = h5py.File(outpath)
        
        self.use_h5py = use_h5py
        
        print >> sys.stderr, 'DlibFaceWorker: ready'
    
    def imread(self, path):
        img = io.imread(path)
        dets = self.detector(img, 1)
        return img, dets
    
    def featurize(self, path, obj):
        img, dets = obj
        for k,d in enumerate(dets):
            shape = self.sp(img, d)
            face_descriptor = self.facerec.compute_face_descriptor(img, shape, 10)
            
            if self.use_h5py:
                self.db['%s/%d/feat' % (os.path.basename(path), k)] = np.array(face_descriptor)
                self.db['%s/%d/img' % (os.path.basename(path), k)] = img[d.top():d.bottom(),d.left():d.right()]
            else:
                print '\t'.join((
                    path, 
                    str(k),
                    '\t'.join(map(str, [d.top(),d.bottom(),d.left(),d.right()])), 
                    '\t'.join(map(str, face_descriptor))
                ))
                sys.stdout.flush()
    
    def close(self):
        if self.use_h5py:
            self.db.close()
