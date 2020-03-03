#!/bin/bash

echo "INCEPTION V3"
./build/internal/bin/classification -f tensorflow -m models/tensorflow/inception_v3_2018_04_27/inception_v3.pb \
		-g 299,299 -i tests/images/cat.jpg -l models/tensorflow/inception_v3_2018_04_27/labels.txt -s 0.0039 -w0,0,0

echo "INCEPTION V4"
./build/internal/bin/classification -f tensorflow -m models/tensorflow/inception_v4_2018_04_27/inception_v4.pb \
		-g 299,299 -i tests/images/cat.jpg -l models/tensorflow/inception_v3_2018_04_27/labels.txt -s 0.0039 -w0,0,0

echo "RESNET v2"
./build/internal/bin/classification -f tensorflow -m models/tensorflow/resnet_v2_101/resnet_v2_101_299_frozen.pb \
	-g 299,299 -i tests/images/cat.jpg -l models/tensorflow/inception_v3_2018_04_27/labels.txt -s 0.0039 -w0,0,0

echo "MOBILENET v1"
./build/internal/bin/classification -f tensorflow -m models/tensorflow/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb  -g 224,224 -i tests/images/cat.jpg -l models/tensorflow/inception_v3_2018_04_27/labels.txt -s 0.017

echo "MOBILENET V2"
./build/internal/bin/classification -f tensorflow -m models/tensorflow/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb  -g 224,224 -i tests/images/cat.jpg -l models/imagenet_slim_labels.txt -s 0.0078 -w 128,128,128 

echo "SQUEEZENET" 
./build/internal/bin/classification -f tensorflow -m models/tensorflow/squeezenet_2018_04_27/squeezenet.pb \
		-g 224,224 -i tests/images/cat.jpg -l models/tensorflow/squeezenet_2018_04_27/labels.txt -s 0.0039 -w0,0,0

echo "RESNET 50"
./build/internal/bin/classification -f tensorflow -m models/tensorflow/resnet_v1_1.0_224/frozen_resnet50v1.pb \
		-g 224,224 -i tests/images/cat.jpg -l models/synset_words.txt -s 1.f -w0,0,0

