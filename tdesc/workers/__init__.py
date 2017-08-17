import sys

from base import BaseWorker

try:
    from dlib_worker import DlibFaceWorker
    from dlib_batch_worker import DlibFaceBatchWorker
except:
    print >> sys.stderr, 'cannot load dlib workers'

from vgg16_worker import VGG16Worker
from yolo_worker import YoloWorker