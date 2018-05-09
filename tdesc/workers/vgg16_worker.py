#!/usr/bin/env python

"""
    vgg16_worker.py
"""

from __future__ import print_function

import sys
import urllib
import contextlib
import cStringIO
import numpy as np

from base import BaseWorker

def import_vgg16():
    global VGG16
    global Model
    global image
    global preprocess_input
    global K
    from keras.applications import VGG16
    from keras.models import Model
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    from keras import backend as K


class VGG16Worker(BaseWorker):
    """
        compute late VGG16 features

        either densely connected (default) or crow (sum-pooled conv5)
    """
    def __init__(self, crow, target_dim=224):
        import_vgg16()
        if crow:
            self.model = VGG16(weights='imagenet', include_top=False)
        else:
            whole_model = VGG16(weights='imagenet', include_top=True)
            self.model = Model(inputs=whole_model.input, outputs=whole_model.get_layer('fc2').output)
            
        self.target_dim = target_dim
        self.crow = crow
        
        self._warmup()
        print('VGG16Worker: ready', file=sys.stderr)
        
    def _warmup(self):
        _ = self.model.predict(np.zeros((1, self.target_dim, self.target_dim, 3)))
        
    def imread(self, path):
        if 'http' == path[:4]:
            with contextlib.closing(urllib.urlopen(path)) as req:
                local_url = cStringIO.StringIO(req.read())
            img = image.load_img(local_url, target_size=(self.target_dim, self.target_dim))
        else:
            img = image.load_img(path, target_size=(self.target_dim, self.target_dim))
            
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
        
    def featurize(self, path, img, return_feat=False):
        feat = self.model.predict(img).squeeze()
        if self.crow:
            feat = feat.sum(axis=(0, 1))
            
        if not return_feat:
            print('\t'.join((path, '\t'.join(map(str, feat)))))
            sys.stdout.flush()
        else:
            return path, feat
            
    def close(self):
        print('VGG16Worker: terminating', file=sys.stderr)
        pass
