# Operator Schemas
This documentation describes the operator definitions.
* [BatchNorm](#batchnorm)
* [Concat](#concat)
* [ConstOp](#constop)
* [Convolution](#convolution)
* [Dropout](#dropout)
* [Eltwise](#eltwise)
* [Fully_connected](#fully_connected)
* [Input_op](#input_op)
* [Pooling](#pooling)
* [PReLu](#prelu)
* [ReLu](#relu)
* [Scale](#scale)
* [Slice](#slice)
* [Softmax](#softmax)


## BatchNorm
BatchNorm operator carries out batch normalization, only for inference phase.

**Inputs**:
* `input`: float32

    the input 4-dimensional tensor of shape NCHW
* `gamma`: float32
* `beta`: float32
    
    the bias as 1-dimensional tensor of size C(Channel)
* `mean`: float32
    
    the estimated mean as 1-dimensional tensor of size C(Channel)
* `var`: float32
    
    the estimated variance as 1-dimensional tensor of size C(Channel)

**Outputs**:
* `output`: float32

**Parameters**:
* `caffe_flavor`: int 
    
    if use caffe version batch_normalization. Default set to 1.
* `rescale_factor`: float32
   

## Concat
Concatenate a list of input tensors into a single output tensor.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `axis`: int 
    
    which axis to concat on, default is set to 1.
    
## ConstOp
A Constant tensor.

**Inputs**:

None

**Outputs**:

* `output`: float32



## Convolution

It computes the output of convolution of an input tensor and filter kernel.

**Inputs**:
* `input`: float32
* `weight`: float32
* `bias`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `kernel_h`: int

    kernel size height
* `kernel_w`: int
* 
    kernel size width
* `stride_h`: int

    stride size height
* `stride_w`: int
* 
    stride size width
* `pad_h`: int

    pad size height
* `pad_w`: int

    pad size width
* `dilation_h` :int
* `dilation_w` :int
* `output_channel`: int

    number of output channel (number of kernel)
* `group`: int
    
## Dropout

Dropout operator for inference phase is Y=X.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

## Eltwise
Compute elementwise operations, such as max or sum, along multiple input tensors.

**Inputs**:
* `input1`: float32
* `input2`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `type`: enum (MAX, SUM)
 

## Fully_connected

Fully-connnected computes the results of X*W+b with X as input,W as weight and b as bias.

**Inputs**:
* `input`: float32
* `weight`: float32
* `bias`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `num_output`: int

    number of output, which is the size of bias


## Input_op
Inpute operator to feed data into network

**Inputs**:

None

**Outputs**:

* `output`: float32


## Lrn

Local Response Normalization normalizes over local input regions.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `local_size`: int
* `norm_region`: int
* `alpha`: float
* `beta`: float
* `k`: float
   

## Pooling

Pooling takes input tensor and applies pooling according to the kernel sizes, stride sizes, pad sizes and pooling types.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `alg`: enum 

    pooling type: *kPoolMax*, *kPoolAvg*
* `global`: int

    if use global pooling
* `caffe_flavor`: int

* `kernel_shape`: a list of int 

    the size of the kernel along each axis (H, W)
* `strides`: a list of int

    stride along each axis (H, W). 
* `pads`: a list of int

    pads zero for each axis (x1_begin, x2_begin...x1_end, x2_end,...). In case of input  of shape NCHW, the pads is (pad_top,pad_left,pad_bottom,pad_right)
    
## PReLu
ReLu(Parameterized Rectified Linear Unit) takes one input data (Tensor) and produces one output data (Tensor) throught `yi=max(0,xi)+slope_i*min(0,xi)` with slopes for negative parts.

**Inputs**:
* `input`: float32
* `slope`: float32

**Outputs**:
* `output`: float32
    

## ReLu
Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, y = max(0, x), is applied to the tensor elementwise.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

## Scale

Scale operator computes the output as scaling the input Y=gamma*X+(bias).

**Inputs**:
* `input`: float32
* `gamma`: float32
* `beta`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `axis`: int 
    
    which axis to coerce the input into 2D, default is set to 1.
* `num_axes`: int

    default set to 1
* `bias_term`: int

    default set to 0

## Slice
Slice op takes an input and slices it along either the num or channel dimension, outputting multiple sliced tensors. 

**Inputs**:
* `input`: float32

**Outputs**:
* `output1`: float32
* `output2`: float32

**Parameters**:
* `axis`: int 
    
    which axis to slice along, default is set to 1 (Channel).
    


## Softmax
Softmax computes the softmax normalized values. The output tensor has the same shape of the input shape.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `axis`: int 
    
    which axis to coerce the input into 2D, default is set to 1.
    

