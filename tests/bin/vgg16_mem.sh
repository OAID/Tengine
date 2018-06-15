#!/bin/bash 

export REPEAT_COUNT=10
export FREE_CONV_KERNEL=1
export FREE_FC_WEIGHT=1

./build/tests/bin/vgg16 ./models/ ./tests/images/bike.jpg
