#!/bin/bash

./build/tools/bin/test_accuracy ./models/mobilenet_deploy.prototxt \
                                ./models/mobilenet.caffemodel \
                                104.007,116.669,122.679 \
                                0.017,0.017,0.017 \
                                data \
                                prob \
                                224,224 \
                                /home/usr/val/                       



