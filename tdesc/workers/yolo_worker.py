#!/usr/bin/env python

"""
    yolo_worker.py
"""

import os
import sys
import urllib
import cStringIO
import numpy as np
from PIL import Image

def import_yolo():
    global DarknetObjectDetector
    from libpydarknet import DarknetObjectDetector

def format_image(pil_image, resize=True, net_size=416):
    pil_image = pil_image.convert('RGB')
    
    if resize:
        pil_image = pil_image.resize((net_size, net_size), Image.BILINEAR)
    
    data = np.array(pil_image).transpose([2,0,1]).astype(np.uint8).tostring()
    return data, (pil_image.size[0], pil_image.size[1])

class DetBBox(object):
    
    def __init__(self, bbox):
        self.left = bbox.left
        self.right = bbox.right
        self.top = bbox.top
        self.bottom = bbox.bottom
        self.confidence = bbox.confidence
        self.cls = bbox.cls

class Darknet_ObjectDetector():
    
    def __init__(self, spec, weight, thresh=0.5, nms=0.4, draw=0):
        self._detector = DarknetObjectDetector(spec, weight, thresh, nms, draw)
    
    def detect_object(self, data, size):
        res = self._detector.detect_object(data, size[0], size[1], 3)
        out = [DetBBox(x) for x in res.content], res.load_time, res.pred_time
        # if self.py_resize:
            # print >> sys.stderr, "!! BBOX is in transformed dimensions -- need to implement fix"
            # pass
        
        return out
    
    @staticmethod
    def set_device(gpu_id):
        DarknetObjectDetector.set_device(gpu_id)



# --

class YoloWorker(object):
    def __init__(self, args, target_dim=224):
        import_yolo()
        
        Darknet_ObjectDetector.set_device(0)
        self.det = Darknet_ObjectDetector(args.cfg_path, args.weight_path, args.thresh, args.nms, 0)
    
    def imread(self, path):
        if 'http' == path[:4]:
            img = cStringIO.StringIO(urllib.urlopen(path).read())
            img = image.load_img(img, target_size=(self.target_dim, self.target_dim))
        else:
            img = image.load_img(path, target_size=(self.target_dim, self.target_dim))
        
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    
    def _warmup(self):
        _ = self.model.predict(np.zeros((1, self.target_dim, self.target_dim, 3)))
    
    def featurize(self, path, img):
        feat = self.model.predict(img).squeeze()
        if self.crow:
            feat = feat.sum(axis=(0, 1))
        
        print '\t'.join((path, '\t'.join(map(str, feat))))
        sys.stdout.flush()
    
    def close(self):
        pass


class DlibFaceWorker(object):
    """ 
        compute dlib face descriptors 
    """
    
    def __init__(self, outpath, use_h5py=False):
        import_dlib()
        self.detector = dlib.get_frontal_face_detector()
        
        ppath = os.path.dirname(os.path.realpath(__file__))
        
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
