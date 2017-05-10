### tdesc

Image processing tools

#### VGG16

```

    # FC2 features
    cat filenames | ~/projects/tdesc/runner.py --model vgg16 > vgg16.descriptors
    
    # CROW features (sum pooled conv5 features)
    cat filenames | ~/projects/tdesc/runner.py --model vgg16 --crow > vgg16.descriptors
```

#### `dlib` face descriptors

```
   
    cat filenames | ~/projects/tdesc/runner.py --model dlib_face --outpath faces.h5
```