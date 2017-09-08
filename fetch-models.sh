#!/bin/bash

# fetch-models.sh
# 
# Download required models

echo "fetch-models.sh: downloading dlib models"
mkdir -p ~/.tdesc/models/dlib
cd ~/.tdesc/models/dlib

wget http://dlib.net/files/mmod_human_face_detector.dat.bz2
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 *bz2


echo "fetch-models.sh: downloading places models"
mkdir -p ~/.tdesc/models/places
cd ~/.tdesc/models/places

wget https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt
wget http://places2.csail.mit.edu/models_places365/whole_resnet18_places365.pth.tar
wget http://places2.csail.mit.edu/models_places365/whole_resnet50_places365.pth.tar