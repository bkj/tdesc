#!/usr/bin/env python

"""
    dlib_worker.py

    Run `dlib` face featurization

    !! Could do a better job w/ batching
"""

import os
import sys

from base import BaseWorker


def import_dlib():
    global dlib
    global h5py
    global io
    global color
    import dlib
    from skimage import io
    from skimage import color


class DlibFaceWorker(BaseWorker):
    """
        compute dlib face descriptors
    """

    def __init__(self, num_jitters=10, dnn=False, det_threshold=0.0, upsample=0):
        import_dlib()
        
        ppath = os.path.join(os.environ['HOME'], '.tdesc')
        
        if not dnn:
            self.detector = dlib.get_frontal_face_detector()
        else:
            detpath = os.path.join(ppath, 'models/dlib/mmod_human_face_detector.dat')
            self.detector = dlib.face_detection_model_v1(detpath)
            
        shapepath = os.path.join(ppath, 'models/dlib/shape_predictor_68_face_landmarks.dat')
        self.sp = dlib.shape_predictor(shapepath)
        
        facepath = os.path.join(ppath, 'models/dlib/dlib_face_recognition_resnet_model_v1.dat')
        self.facerec = dlib.face_recognition_model_v1(facepath)
        
        self.num_jitters = num_jitters
        self.dnn = dnn
        self.det_threshold = det_threshold
        self.upsample = upsample
        
        print >> sys.stderr, 'DlibFaceWorker: ready (dnn=%d | num_jitters=%d)' % (int(dnn), int(num_jitters))

    def imread(self, path):
        img = io.imread(path)
        if img.shape[-1] == 4:
            img = color.rgba2rgb(img)
        elif len(img.shape) == 2:
            img = color.grey2rgb(img)

        if not self.dnn:
            dets, _, _ = self.detector.run(img, self.upsample, self.det_threshold)
        else:
            dets = []

        return img, dets

    def featurize(self, path, obj, return_feat=False):
        img, dets = obj
        if self.dnn:
            dets, _ = self.detector(img)

        feats = []
        for k,d in enumerate(dets):
            shape = self.sp(img, d)
            face_descriptor = self.facerec.compute_face_descriptor(img, shape, self.num_jitters)

            if not return_feat:
                print '\t'.join((
                    path,
                    str(k),
                    '\t'.join(map(str, [d.top(),d.bottom(),d.left(),d.right()])),
                    '\t'.join(map(str, face_descriptor))
                ))
                sys.stdout.flush()
            else:
                feats.append({
                    "k" : k,
                    "bbox" : [d.top(),d.bottom(),d.left(),d.right()],
                    "desc" : face_descriptor
                })

        if return_feat:
            return path, feats

    def close(self):
        print >> sys.stderr, 'DlibFaceWorker: terminating'
