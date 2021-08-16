# Technical Highlights

**Tengine** is led by **[OPEN AI LAB](http://www.openailab.com)** .this project meets the requirements of **Fast**、**Efficient** deployment of deep learning neural network model on embedded devices. In order to realize cross-platform deployment in many **AIoT** applications, this project is reconstructed based on the original Tengine project **C Language** and the deep framework is cut according to the limited resources of embedded devices. At the same time, a completely separated front-end design is adopted, which is beneficial to the rapid transplantation and deployment of heterogeneous computing units such as CPU, GPU and NPU, and reduces the cost of evaluation and migration.

## Multiple Hardware Support

Tengine supports a variety of hardware backend to accelerate reasoning of neural network model, including（ARM、X86、MIPS、RISC-V）、GPU（Mail、NV、AMD、Adreno、PowerVR）、NPU（VSI、NNIE、DLA）。

## High-performance

By providing calculation graph optimization (operator combination and operator removal), the structure of the native network model is optimized and the calculation amount is reduced.

At the same time, according to different CPU architectures, fine manual compilation is adopted to realize the extreme optimization of the Kernel with high computational requirements, and give full play to the peak computing power of hardware. 

## Heterogeneous Cut Graph

In order to support various computing units on different SoCs, after loading the model, Tengine obtains the acceleration characteristics of the currently specified hardware, flexibly divides the original computing diagram, makes full use of computing power, and improves the generalization of model support.

## Quantitative Support

Support the two mainstream quantization strategies (symmetric sub-channel quantization and asymmetric layered quantization) to achieve the goals of low-bit compression and performance acceleration of the model, and at the same time seamlessly interface with mainstream NPU acceleration engine. A compensation scheme for low bit quantization accuracy is provided.

## Mixing Precision

In order to give full play to the hardware computing resources and ensure the reasoning accuracy of the model, the mixed precision computing mode is supported.

## Lightweight Deployment

The latest Tengine core module code is developed in c language, which is independent of third-party library. the minimum executable static library size is less than 100KB, and it can even be deployed on mainstream MCU.
