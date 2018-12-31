# Operator Schemas
This documentation describes the operator definitions.
- [Operator Schemas](#operator-schemas)
    - [BatchNorm](#batchnorm)
    - [Concat](#concat)
    - [ConstOp](#constop)
    - [Convolution](#convolution)
    - [Deconvolution](#deconvolution)
    - [Detection_output](#detection_output)
    - [Dropout](#dropout)
    - [Eltwise](#eltwise)
    - [Flatten](#flatten)
    - [Fully_connected](#fully_connected)
    - [Input_op](#input_op)
    - [LRN](#lrn)
    - [LSTM](#lstm)
    - [Normalize](#normalize)
    - [Permute](#permute)
    - [Pooling](#pooling)
    - [Priorbox](#priorbox)
    - [PReLu](#prelu)
    - [Region](#region)
    - [Resize](#resize)
    - [Reorg](#reorg)
    - [Reshape](#reshape)
    - [ReLu](#relu)
    - [RPN](#rpn)
    - [Roi_pooling](#roi_pooling)
    - [Scale](#scale)
    - [Slice](#slice)
    - [Softmax](#softmax)


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
* `activation`: int

## Deconvolution

Deconvolution operator multiplies each input value by a kernel elementwise, and sums over the resulting on output windows.


**Inputs**:
* `input`: float32
* `weight`: float32
* `bias`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `kernel_size`: int

* `stride`: int

    stride size 

* `pad`: int

    pad size 

* `num_output`: int

    number of output channel (number of kernel)

* `dilation`: int


## Detection_output

Detection_output operator used in SSD-detection network.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32
    which is coordinates of detected boxes and corresponding confidences for each class

**Parameters**:
* `num_classes`: int

    number of classes of detection benchmark (21 for VOC and 81 for COCO)

* `confidence_threshold`: float
* `nms_threshold`: float
* `keep_top_k`: int

    num of top_k keeping for results of nms

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
 
## Flatten
The Flatten operator is a utility op that flattens an input of shape [n, c, h, w] to a simple vector output of shape [n, (c*h*w),1, 1].

**Inputs**:
* `input1`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `axis `: int

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


## LRN

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

## LSTM

LSTM operator.

**Inputs**:
* `input`: float32
* `kernel`: float32
* `bias`: float32,(optional)
* `init_c`: float32,(optional)
* `init_h`: float32,(optional)

**Outputs**:
* `output`: float32

**Parameters**:
* `forget_bias`: float32
* `has_peephole`: bool
* `has_projection`: bool
* `input_size`: int32
* `hidden_size`: int32
* `cell_size`: int32
* `output_len`: int32


## Normalize

Normalize operator normalizes the input alone channel axis with L2 normalization, used in SSD-detection network

**Inputs**:
* `input`: float32
* `scale`: float32

**Outputs**:
* `output`: float32

## Permute

Permute operator permutes the input with specific order, used in SSD-detection network.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `order0`: int
* `order1`: int
* `order2`: int
* `order3`: int

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

## Priorbox

Priorbox operator computes the prior boxes for SSD (single shot detection) network. It will compute the prior boxes according to the original image size or specific image size defined by proto, as well as according to other parameters: max box size, min box size, aspect ratio for box etc.

**Inputs**:
* `input`: float32
* `image_width`:int32
* `image_height`:int32

**Outputs**:
* `output`: float32

**Parameters**:
* `min_size`: float32
* `max_size`: float32
* `variance`: float32
* `aspect_ratio`: float32
* `flip`: int32
* `clip`: int32
* `img_size`: int32
* `img_h`: int32
* `img_w`: int32

## PReLu
ReLu(Parameterized Rectified Linear Unit) takes one input data (Tensor) and produces one output data (Tensor) through `yi=max(0,xi)+slope_i*min(0,xi)` with slopes for negative parts.

**Inputs**:
* `input`: float32
* `slope`: float32

**Outputs**:
* `output`: float32
    
## Region
Region operator is used in YOLO network. It is a post process for the network output to rescale the output into [0,1]  for compute the final  detection out boxes.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `num_classes`: int32
* `side`: int32
* `num_box`: int32
* `coords`: int32
* `confidence_threshold`: float32
* `nms_threshold`: float32
* `biases`: float32

## Resize
Resize layer used in several networks for upsampling. Current resize op supports two resize methods: bilinear_resize and nearest_neighbor_resize.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `type`: 0 for NEAREST_NEIGHBOR, 1 for BILINEAR_RESIZE
* `scale_w`: float32, `scale_w = out_w/in_w`
* `scale_h`: float32, `scale_h = out_h/in_h`

## Reorg
Reorg operator is used in YOLO network. It is a process for the network to re-organize the data according the stride.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `stride`: int32

## Reshape
The Reshape operator can be used to change the dimensions of its input, without changing its data. 

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `dim_0`: int32
* `dim_1`: int32
* `dim_2`: int32
* `dim_3`: int32

## ReLu
Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, y = max(0, x), is applied to the tensor elementwise.

**Inputs**:
* `input`: float32

**Outputs**:
* `output`: float32

**Parameters**:
* `negative_slope`: float

    the relu with negative_slope = 0.1, is also called leaky-activation

## RPN
Region Proposal Network(RPN) operator used in Faster-RCNN network. It generates proposal anchors.

**Inputs**:
* `input0`: float32 
	scoretensor

* `input1`: float32
    featmap tensor

**Outputs**:
* `output`: float32

**Parameters**:
* `nms_thresh`: float32

* `post_nms_topn`: int
 
    postprocess_nms_topn

* `per_nms_topn`: int
    preprocess_nms_topn

* `min_size`: int
* `basesize`: int
* `feat_stride`: int
* `anchor_scales`: <float>

    a list of anchor scales

* `ratios`: <float>

    a list of ratios

## Roi_pooling
Roi_pooling operator used in Faster-RCNN network. It performs max pooling on regions of interest(ROI) specified by input.

**Inputs**:
* `input0`: float32 

    input0 is [N x C x H x W] feature maps on which pooling is performed.
* `input1`: float32

    Input[1] [ R x 4] contains a list R ROI with each 4 coordinates.

**Outputs**:
* `output`: float32

**Parameters**:
* `pooled_h`: int

    The pooled output height.
* `pooled_w`: int

    The pooled output width.

* `spatial_scale`: float
    Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.

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
    

