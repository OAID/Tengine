# Imagenet classification with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is the implementation of imagenet classification with [Tengine](https://github.com/OAID/Tengine).

It can be used to test the following imagenets:

- squeezenet_v1.1
- mobilenet_v1
- mobilenet_v2
- resnet50
- googlenet
- inception_v4
- inception_v3
- alexnet
- vgg16

## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb).

Store these files into `${Tengine_ROOT}/models/`

- squeezenet_v1.1(already existed in Tengine_ROOT/models/)
- mobilenet_v1(already existed in Tengine_ROOT/models/)
- mobilenet_v2
  - mobilenet_v2_deploy.prototxt
  - mobilenet_v2.caffemodel
- resnet50
  - resnet50.prototxt
  - resnet50.caffemodel
- googlenet
  - googlenet.prototxt
  - googlenet.caffemodel
- inception_v4
  - inception_v4.prototxt
  - inception_v4.caffemodel
- inception_v3
  - deploy_inceptionV3.prototxt
  - deploy_inceptionV3.caffemodel
- alexnet
  - alex_deploy.prototxt
  - alexnet.caffemodel
- vgg16
  - vgg16.prototxt
  - vgg16.caffemodel


## Build examples
```
cd  ${Tengine_ROOT}
make
make install
```
build as ${TENGINE_ROOT}/examples/readme.md

## Run
- go to the directory of the executive program

    ```
    cd ${Tengine_ROOT}/examples/build/imagenet_classification
    ```
- test an image:

    ```
    [Usage]: ./Classify [-n model_name] [-p proto_file] [-m model_file] [-l label_file] [-i image_file]
                        [-g img_h,img_w] [-s scale] [-w mean[0],mean[1],mean[2]] [-r repeat_count]
                        [-d device_name] [-h]
    ```
    Default model name is : `squeezenet`

    Default label file is : `Tengine_ROOT/models/synset_words.txt`

    Default image file is : `Tengine_ROOT/tests/images/cat.jpg`

    Default image size is : `227,227`

    Default scale value is : `1.f`

    Default mean values are : `104.007,116.669,122.679`


- usage examples:

    ```
    ./Classify -h               // display usage
    ./Classify -n squeezenet    // test squeezenet with default cat.jpg
    ./Classify -n mobilenet     // test mobilenet with default cat.jpg
    ./Classify -n mobilenet_v2  // test mobilenet_v2 with default cat.jpg
    ./Classify -n resnet50      // test resnet50 with default cat.jpg
    ./Classify -n alexnet       // test alexnet with default cat.jpg
    ./Classify -n googlenet     // test googlenet with default cat.jpg
    ./Classify -n inception_v3  // test inception_v3 with default cat.jpg
    ./Classify -n inception_v4  // test inception_v4 with default cat.jpg
    ./Classify -n vgg16 -i ~/tengine/tests/images/bike.jpg  // test vgg16 with specified test image
    
    // test squeezenet with specified parameters
    ./Classify -n squeezenet -g 227,227 -s 1.f -w 104.007,116.669,122.679
    
    // test with specified files
    ./Classify -p ~/tengine/models/sqz.prototxt -m ~/tengine/models/squeezenet_v1.1.caffemodel
               -l ~/tengine/models/synset_words.txt -i ~/tengine/tests/images/bike.jpg
    ```

