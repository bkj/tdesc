### tdesc

Image processing tools

#### VGG16

Takes list of filenames, writes filename + descriptor to TSV.

```

    # FC2 features
    cat filenames | ~/projects/tdesc/runner.py --model vgg16 > vgg16.descriptors
    
    # CROW features (sum pooled conv5 features)
    cat filenames | ~/projects/tdesc/runner.py --model vgg16 --crow > vgg16.descriptors
```

#### `dlib` face descriptors

Takes list of filenames, writes
    - extracted face images
    - face descriptors
to `h5py` file.

```

    cat filenames | ~/projects/tdesc/runner.py --model dlib_face --outpath faces.h5
```