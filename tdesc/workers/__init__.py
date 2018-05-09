from __future__ import print_function

import sys

from base import BaseWorker

try:
    from dlib_worker import DlibFaceWorker
    from dlib_batch_worker import DlibFaceBatchWorker
except:
    print('cannot load dlib workers', file=sys.stderr)

try:
    from yolo_worker import YoloWorker
except:
    print('cannot load darknet workers', file=sys.stderr)

try:
    from vgg16_worker import VGG16Worker
except:
    print('cannot load keras workers', file=sys.stderr)

try:
    from places_worker import PlacesWorker
except:
    print('cannot load pytorch workers', file=sys.stderr)