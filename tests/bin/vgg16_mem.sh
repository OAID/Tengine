#!/bin/bash 

export REPEAT_COUNT=10
export LOW_MEM_MODE=1

./build/tests/bin/vgg16 ./models/ ./tests/images/bike.jpg
