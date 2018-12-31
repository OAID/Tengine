# Imagenet classification with Tengine model

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is the implementation of imagenet classification with Tengine model.

It can be used to test the following imagenets:

- squeezenet_V1.1
- mobilenet_V1
- resnet50
- googlenet
- inception_v4
- inception_v3
- alexnet
- vgg16

## Create required tengine model files

Create required tengine model files by using the conversion tool in `${Tengine_ROOT}/examples/tengine_model/convert`.

Store the generated tengine model files into `${Tengine_ROOT}/models/`.

- squeezenet.tmfile
- mobilenet.tmfile
- resnet50.tmfile
- googlenet.tmfile
- inception_v4.tmfile
- inception_v3.tmfile
- alexnet.tmfile
- vgg16.tmfile


## Build examples
```
cd  ${Tengine_ROOT}
make install
```
build as ${TENGINE_ROOT}/examples/readme.md

## Run
- go to the directory of the executive program

    ```
    cd ${Tengine_ROOT}/examples/build/tengine_model/classification
    ```
- test an image:

    ```
    [Usage]: ./tm_classify [-n model_name] [-t tm_file] [-l label_file] [-i image_file]
                           [-g img_h,img_w] [-s scale] [-w mean[0],mean[1],mean[2]] [-r repeat_count] [-h]
    ```
    Default model name is : `squeezenet`

    Default label file is : `Tengine_ROOT/models/synset_words.txt`

    Default image file is : `Tengine_ROOT/tests/images/cat.jpg`

    Default image size is : `227,227`

    Default scale value is : `1.f`

    Default mean values are : `104.007,116.669,122.679`


- usage examples:

    ```
    ./tm_classify -h               // display usage
    ./tm_classify -n squeezenet    // test squeezenet with default cat.jpg
    ./tm_classify -n mobilenet     // test mobilenet with default cat.jpg
    ./tm_classify -n resnet50      // test resnet50 with default cat.jpg
    ./tm_classify -n alexnet       // test alexnet with default cat.jpg
    ./tm_classify -n googlenet     // test googlenet with default cat.jpg
    ./tm_classify -n inception_v3  // test inception_v3 with default cat.jpg
    ./tm_classify -n inception_v4  // test inception_v4 with default cat.jpg
    ./tm_classify -n vgg16 -i ~/tengine/tests/images/bike.jpg  // test vgg16 with specified test image
    
    // test squeezenet with specified parameters
    ./tm_classify -n squeezenet -g 227,227 -s 1.f -w 104.007,116.669,122.679
    
    // test with specified files
    ./tm_classify -t ~/tengine/models/squeezenet.tmfile
                  -l ~/tengine/models/synset_words.txt -i ~/tengine/tests/images/bike.jpg
    ```

