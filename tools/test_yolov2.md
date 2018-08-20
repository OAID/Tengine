# Tengine examples

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb)


## prepare the test images
Download voc_2007_test from [FTP](ftp://ftp.openailab.net/Tengine_models/voc_2007_test.tgz)

```
  tar -zxvf ./voc_2007_test.tgz
  cd voc_2007_test
  python voc_label.py
  mv 2007_test.txt ~/{TENGINE_DIR}/tools/data/
```
## Test yolov2 recall and precision
### 1. Set the TENGINE_DIR in CMakeLists.txt
```
    cd {TENGINE_DIR}/tools/bin
    vim CMakeLists.txt
```
### 2. Set image_list and root_path
```
    vim test_yolov2.cpp
```
### 3. build
``` 
    mkdir build
	cd build
	cmake ..
    make -j4
```
### 4. Run
```
    ./TEST_YOLO
```