import sys

from base import BaseWorker

try:
    from dlib_worker import DlibFaceWorker
    from dlib_batch_worker import DlibFaceBatchWorker
except:
    print >> sys.stderr, 'cannot load dlib workers'

try:
    from yolo_worker import YoloWorker
except:
    print >> sys.stderr, 'cannot load darknet workers'

try:
    from vgg16_worker import VGG16Worker
except:
    print >> sys.stderr, 'cannot load keras workers'

try:
    from places_worker import PlacesWorker
except:
    print >> sys.stderr, 'cannot load pytorch workers'