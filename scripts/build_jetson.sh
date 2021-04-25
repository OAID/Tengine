#!/bin/bash 


mkdir -p build-jetson
pushd build-jetson
cmake                                                                   \
    -DTENGINE_ENABLE_TENSORRT=ON                                        \
    -DCUDA_INCLUDE_DIR=/usr/local/cuda/include                          \
    -DCUDA_LIBRARY_DIR=/usr/local/cuda/targets/aarch64-linux/lib        \
    -DTENSORRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu               \
    -DTENSORRT_LIBRARY_DIR=/usr/lib/aarch64-linux-gnu                   \
    ..
cmake --build . --parallel `nproc`
cmake --build . --target install
popd
