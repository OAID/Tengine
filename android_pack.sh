#!/bin/bash

mkdir -p android_pack

cp build/libtengine.so  ./android_pack  
cp build/libhclcpu.so  ./android_pack
cp install/lib/libc++_shared.so ./android_pack


