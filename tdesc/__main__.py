#!/usr/bin/env python

"""
    tdesc
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--io-threads', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=10)
    
    # VGG16 options
    parser.add_argument('--crow', action="store_true")
    
    # DlibFace options
    parser.add_argument('--outpath', type=str, default='./db.h5')
    
    # Yolo options
    parser.add_argument('--yolo-cfg-path', type=str, required=False)
    parser.add_argument('--yolo-weight-path', type=str, required=False)
    parser.add_argument('--yolo-name-path', type=str, required=False)
    parser.add_argument('--yolo-thresh', type=float, default=0.1)
    parser.add_argument('--yolo-nms', type=float, default=0.3)
    
    args = parser.parse_args()
    
    if args.model == 'yolo':
        assert args.yolo_cfg_path is not None
        assert args.yolo_weight_path is not None
        assert args.yolo_nms is not None
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.model == 'vgg16':
        from tdesc.workers import VGG16Worker
        worker = VGG16Worker(args.crow)
    elif args.model == 'dlib_face':
        from tdesc.workers import DlibFaceWorker
        worker = DlibFaceWorker(args.outpath)
    elif args.model == 'yolo':
        from tdesc.workers import YoloWorker
        worker = YoloWorker(**{
            "cfg_path" : args.yolo_cfg_path,
            "weight_path" : args.yolo_weight_path,
            "name_path" : args.yolo_name_path,
            "thresh" : args.yolo_thresh,
            "nms" : args.yolo_nms,
        })
    else:
        raise Exception()
    
    worker.run(args.io_threads)

