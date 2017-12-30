# Tengine Overview

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

**Tengine**, developed by **OPEN** AI LAB, is a lite, high-performance, and modular inference engine for embedded device.

Tengine is composed of four modules: **core/operator/serializer/executor**.

- [**core**](core)  provides the basic components and functionalities of the system.
- [**operator**](operator)  defines the schema of operators, such as convolution, relu, pooling, etc. al. Here are the current support [operator list](doc/operator_ir.md) 
- [**serializer**](serializer)  is to load the saved model. The serializer framework is extensible to support different format, including the customized one. Current version only support caffe model. Tensorflow and MXNet support will be the next.

- [**executor**](executor) implements the code to run graph and operators. Current version only provides a highly optimized implementation for single A72.

This version can load and run caffe model of **mobilenet** and **squeezenet** directly.  For more details, please goto [**install**](doc/install.md).

`NOTE`: old caffe model has to be upgraded using **upgrade_net_proto_binary/upgrade_net_proto_binary** from caffe's package

## Performance

The data is collected on **single 1.8G A72** on chip RK3399, by repeating calling the forward interface to get the average time cost (ms) per run.


|NN  |Caffe(Openblas)|Tengine|
|----|---------------|-------|
|squeezenet|147|91|
|mobilenet|306|122|


For details to run benchmark, please visit [**benchmark**](doc/benchmark.md) page.

## Build and Install
please refer to the [install](doc/install.md) page


## Develop New Operator
It is easy to add new operator to Tengine. Here is the guide on [new operator](doc/operator_dev.md)

## Support New Model Format

Tengine can be extended to support new serialization format, by building new serializer module. 

[How to build new serializer module](doc/serializer_dev.md)

## Release History

### version 0.1.2 - 2017/12/30

Update documents, as well a few fixes.

### version 0.1.0 - 2017/12/29

Initial release of single A72 support




