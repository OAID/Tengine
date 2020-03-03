#!/bin/bash

cmake -DCMAKE_C_COMPILER=arm-himix200-linux-gcc \
      -DCMAKE_CXX_COMPILER=arm-himix200-linux-g++ \
      -DTENGINE_DIR=/root/work/git/tengine_auto/tengine \
      -DARM=TRUE \
      -DPROTOBUF_DIR=/root/work/deliver/HI3519A/libs/pblib \
      -DBLAS_DIR= \
      ..
