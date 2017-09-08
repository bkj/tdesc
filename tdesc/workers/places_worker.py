#!/usr/bin/env python

"""
    vgg16_worker.py
"""

import sys
import urllib
import contextlib
import cStringIO
import numpy as np

from base import BaseWorker

def import_places():
    global torch
    global Variable
    global transforms
    global Image
    import torch
    from torch.autograd import Variable
    from torchvision import transforms
    from PIL import Image


class PlacesWorker(BaseWorker):
    """
        compute late features of places365 network

        either crow (avg-pooled last conv) or not crow (classification)
    """
    def __init__(self, weight_path, target_dim=256):
        import_places()
        
        self.model = torch.load(weight_path).cuda()
        _ = self.model.eval()
        
        self.prep = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print >> sys.stderr, 'PlacesWorker: ready'
        
    def imread(self, path):
        if 'http' == path[:4]:
            with contextlib.closing(urllib.urlopen(path)) as req:
                local_url = cStringIO.StringIO(req.read())
            
            img = Image.open(local_url)
        else:
            img = Image.open(path)
            
        img = img.convert('RGB')
        return self.prep(img)
    
    def _partial_forward(self, model, x):
        """ resnet forward pass w/o final fc layer """
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        
        x = model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
        
    def featurize(self, path, img, return_feat=False):
        
        img = Variable(img.unsqueeze(0), volatile=True).cuda()
        
        # # Predict classes -- skipping for now
        # logit = model.forward(img)
        # h_x = F.softmax(logit).data.squeeze()
        # probs, idx = h_x.sort(0, True)
        
        # probs = probs.cpu().numpy()
        # idx = idx.cpu().numpy()
        # classes = class_names[idx]
        
        # print dict(zip(classes, probs))
        
        feat = self._partial_forward(self.model, img)
        feat = feat.data.cpu().numpy().squeeze()
        
        if not return_feat:
            print '\t'.join([path] + map(str, feat))
            sys.stdout.flush()
        else:
            return path, feat
        
    def close(self):
        print >> sys.stderr, 'PlacesWorker: terminating'
        pass
