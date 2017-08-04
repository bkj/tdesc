#!/usr/bin/env python

"""
    quick-align.py
"""

import os
import sys
import dlib
import argparse
import numpy as np
from PIL import Image
from time import time
from concurrent.futures import ProcessPoolExecutor

# --
# Load models

print_interval = 200

det = dlib.get_frontal_face_detector()
shapepath = '/home/bjohnson/.tdesc/models/dlib/shape_predictor_68_face_landmarks.dat'
fa = dlib.face_align(shapepath)

# --
# Helper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best-face', action='store_true')
    parser.add_argument('--size', type=int, default=150)
    parser.add_argument('--padding', type=float, default=0.25)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--outdir', type=str, default='./face_chips/')
    parser.add_argument('--strip-path', type=int, default=2)
    return parser.parse_args()

def do_work(path, outdir, best_face=False, size=150, padding=0.25, threshold=0.0, strip_path=2):
    try:
        img = np.array(Image.open(path).convert('RGB'))
        rects, confs, _ = det.run(img, adjust_threshold=threshold)
        
        for k,(rect,conf) in enumerate(zip(rects, confs)):
            
            dirname = os.path.dirname(path)
            # >>
            # !! Drop prefix
            dirname = '/'.join(dirname.split('/')[strip_path:])
            # <<
            dirname = os.path.join(outdir, dirname)
            if not os.path.exists(dirname):
                try:
                    os.makedirs(dirname)
                except:
                    pass
            
            if best_face:
                fname = os.path.basename(path)
            else:
                fname = os.path.basename(path).split('.')[0] + '-%s' % k + '.jpg'
            
            fa.save(img, rect, os.path.join(dirname, fname), size=size, padding=padding)
            
            print '\t'.join((
                path, 
                str(k),
                '\t'.join(map(str, [rect.top(),rect.bottom(),rect.left(),rect.right()])), 
                str(conf)
            ))
            sys.stdout.flush()
            
            if best_face:
                break
    except KeyboardInterrupt:
        raise
    except:
        print >> sys.stderr, "do_work: error at %s" % path
    
    return path

# --

if __name__ == "__main__":
    args = parse_args()
    gen = (line.strip() for line in sys.stdin)
    i = 0
    start_time = time()
    
    def f(x):
        return do_work(
            x, 
            outdir=args.outdir,
            best_face=args.best_face,
            size=args.size,
            padding=args.padding,
            threshold=args.threshold,
            strip_path=args.strip_path,
        )
    
    with ProcessPoolExecutor(max_workers=16) as execr:
        res = execr.map(f, gen)
        for r in res:
            i += 1
            if not i % print_interval:
                print >> sys.stderr, "%d images | %f seconds " % (i, time() - start_time)
            sys.stdout.flush()