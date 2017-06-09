## tdesc

Image processing tools

### Setup

```
python setup.py install
```

### Models

#### VGG16

Takes list of filenames, writes filename + descriptor to TSV.

```

    # FC2 features
    cat filenames | python -m tdesc --model vgg16 > vgg16-fc2.descriptors
    
    # CROW features (sum pooled conv5 features)
    cat filenames | python -m tdesc --model vgg16 --crow > vgg16-crow.descriptors
```

#### `dlib` face descriptors

Takes list of filenames, writes filename + descriptor to TSV.

```

    cat filenames | python -m tdesc --model dlib_face --outpath faces.h5
```

It seems like the `AVX_INSTRUCTIONS` option in `dlib` makes a big difference (8x on my box?).

### Details

#### Threaded IO

The threaded IO can give a pretty big speedup.

These are benchmarks on my system, reading images over the network.  The boost you get from this depends on lots of factors.

```

    # Naive IO
    cat urls | ./naive-runner.py --crow > /dev/null

        Using Theano backend.
        Using cuDNN version 5110 on context None
        Mapped name None to device cuda0: TITAN X (Pascal) (0000:04:00.0)
        VGG16Worker: ready
        0 images | 0.055656 seconds 
        100 images | 4.867988 seconds 
        200 images | 10.684320 seconds 
        300 images | 14.986690 seconds 
        400 images | 19.420027 seconds 
        500 images | 25.329047 seconds 
        600 images | 30.432902 seconds 
        700 images | 35.608703 seconds 
        800 images | 40.109027 seconds 
        900 images | 44.767013 seconds 
        999 images | 53.446249 seconds 

    
    # One thread
    cat urls | python -m tdesc --crow --io-threads 1 > /dev/null
    
        Using Theano backend.
        Using cuDNN version 5110 on context None
        Mapped name None to device cuda0: TITAN X (Pascal) (0000:04:00.0)
        VGG16Worker: ready
        100 images | 4.254059 seconds
        200 images | 7.766741 seconds
        300 images | 10.954665 seconds
        400 images | 14.537050 seconds
        500 images | 18.294582 seconds
        600 images | 23.202393 seconds
        700 images | 28.520624 seconds
        800 images | 32.765243 seconds
        900 images | 36.555996 seconds
        1000 images | 41.932386 seconds
    
    
    # Two threads
    cat urls | python -m tdesc --crow --io-threads 2 > /dev/null
    
        ...
        900 images | 15.679864 seconds
        1000 images | 17.494891 seconds
    
    # Four threads
    cat urls | python -m tdesc --crow --io-threads 4 > /dev/null
        
        ...    
        900 images | 7.591560 seconds
        1000 images | 8.437886 seconds
```

#### Batching images

Might be able to get a (modest) speedup by predicting on batches of images, instead of predicting one-by-one.
