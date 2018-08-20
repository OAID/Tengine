#!/bin/bash

./build/tools/bin/test_accuracy ./models/sqz.prototxt \
                                ./models/squeezenet_v1.1.caffemodel \
                                104.007,116.669,122.679 \
                                1,1,1 \
                                data \
                                prob \
                                227,227 \
                                /home/firefly/val/                       



