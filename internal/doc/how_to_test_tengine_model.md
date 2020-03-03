# How to test tengine model

This guide gives instructions on how to test tengine model on your system.

## 1. Build Tengine with tengine serializer

* Prepare config file

Verify `CONFIG_TENGINE_SERIALIZER` option in makefile.config is valid.
```
CONFIG_TENGINE_SERIALIZER=y
```

* Build the Tengine
```
make
```

## 2. Prepare necessary model files

Store following necessary model and label files in `${Tengine_ROOT}/models/`.

- sqz.prototxt
- squeezenet_v1.1.caffemodel
- mobilenet_deploy.prototxt
- mobilenet.caffemodel
- resnet50.prototxt
- resnet50.caffemodel
- googlenet.prototxt
- googlenet.caffemodel
- inception_v4.prototxt
- inception_v4.caffemodel
- deploy_inceptionV3.prototxt
- deploy_inceptionV3.caffemodel
- alex_deploy.prototxt
- alexnet.caffemodel
- vgg16.prototxt
- vgg16.caffemodel
- synset_words.txt
- synset2015.txt

## 3. Run test program of tengine model

Test program of tengine model `test_tm` can be used to test squeezenet, mobilenet, alexnet, googlenet, inception_v3, inception_v4, resnet50 and vgg16.
In `test_tm`, it loads caffe models and saves them into tengine models, then uses generated tengine models to run inference.
Please run `test_tm` in the root directory of tengine, and check the results.

```
cd ~/tengine
mkdir tengine_models                       // this directory is used to store generated tengine model files

./build/tests/bin/test_tm -m squeezenet    // test squeezenet
./build/tests/bin/test_tm -m mobilenet     // test mobilenet
./build/tests/bin/test_tm -m alexnet       // test alexnet
./build/tests/bin/test_tm -m googlenet     // test googlenet
./build/tests/bin/test_tm -m inception_v3  // test inception_v3
./build/tests/bin/test_tm -m inception_v4  // test inception_v4
./build/tests/bin/test_tm -m vgg16         // test vgg16
./build/tests/bin/test_tm -m resnet50 -p ./tests/images/bike.jpg  // test resnet50
```

Output message:
* Squeezenet
```
".2763 - "n02123045 tabby, tabby cat
".2673 - "n02123159 tiger cat
".1766 - "n02119789 kit fox, Vulpes macrotis
".0827 - "n02124075 Egyptian cat
".0777 - "n02085620 Chihuahua
```

* Mobilenet
```
".5976 - "n02123159 tiger cat
".9550 - "n02119022 red fox, Vulpes vulpes
".8679 - "n02119789 kit fox, Vulpes macrotis
".4274 - "n02113023 Pembroke, Pembroke Welsh corgi
".3646 - "n02123045 tabby, tabby cat
```

* Alexnet
```
".3094 - "n02124075 Egyptian cat
".1761 - "n02123159 tiger cat
".1221 - "n02123045 tabby, tabby cat
".1132 - "n02119022 red fox, Vulpes vulpes
".0421 - "n02085620 Chihuahua
```

* Googlenet
```
".5009 - "n02123159 tiger cat
".2283 - "n02123045 tabby, tabby cat
".1612 - "n02124075 Egyptian cat
".0283 - "n02127052 lynx, catamount
".0134 - "n02123394 Persian cat
```

* Inception_v3
```
".0946 - "n02123159 tiger cat
".0549 - "n02123045 tabby, tabby cat
".0404 - "n02124075 Egyptian cat
".0136 - "n02127052 lynx, catamount
".0081 - "n02119789 kit fox, Vulpes macrotis
```

* Inception_v4
```
".7556 - "n02123159 tiger cat
".0867 - "n02123045 tabby, tabby cat
".0676 - "n02124075 Egyptian cat
".0040 - "n02127052 lynx, catamount
".0035 - "n02123394 Persian cat
```

* Vgg16
```
".2525 - "n02119789 kit fox, Vulpes macrotis
".2219 - "n02119022 red fox, Vulpes vulpes
".1435 - "n02124075 Egyptian cat
".1371 - "n02123159 tiger cat
".1078 - "n02123045 tabby, tabby cat
```

* Resnet50
```
".9239 - "n03792782 mountain bike, all-terrain bike, off-roader
".0133 - "n03208938 disk brake, disc brake
".0127 - "n02835271 bicycle-built-for-two, tandem bicycle, tandem
".0060 - "n04557648 water bottle
".0044 - "n03786901 mortar
```

