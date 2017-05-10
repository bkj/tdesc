#!/bin/bash

# fetch-models.sh
# 
# Download required models

echo "fetch-models.sh: downloading dlib models"
wget http://dlib.net/files/mmod_human_face_detector.dat.bz2
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

mkdir -p models/dlib
mv *bz2 models/dlib
cd models/dlib
bunzip2 *bz2
