
#!/usr/bin/env python

"""
    workers.py
"""

import os
import sys
import urllib
import cStringIO
import numpy as np

from keras.applications import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

class VGG16Worker(object):
    
    def __init__(self, crow, target_dim=224):
        if crow:
            self.model = VGG16(weights='imagenet', include_top=False)
        else:
            whole_model = VGG16(weights='imagenet', include_top=True)
            self.model = Model(inputs=whole_model.input, outputs=whole_model.get_layer('fc2').output)
        
        self.target_dim = target_dim
        self.crow = crow
        
        self._warmup()
    
    def _warmup(self):
        _ = self.model.predict(np.zeros((1, self.target_dim, self.target_dim, 3)))
    
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
    
    def featurize(self, path, img):
        feat = self.model.predict(img)
        if self.crow:
            feat = feat.sum(axis=(0, 1))
        else:
            feat = feat.squeeze()
        
        print '\t'.join((path, '\t'.join(map(str, feat))))
    
    def close(self):
        pass