## Install dependent libraries
### Protobuf version 3.10.0+ required.
- For loading caffe model or TensorFlow model.
``` shell
sudo apt install libprotobuf-dev protobuf-compiler
```

- If use the Fedora/CentOS ,use follow command instead.
``` shell
sudo dnf install protobuf-devel
sudo dnf install boost-devel glog-devel
```

- ONNX
``` shell
./tools/convert_tool/convert_tool -f onnx -m mobilenet.onnx -o mobilenet.tmfile
```