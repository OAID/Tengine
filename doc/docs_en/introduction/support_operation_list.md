# Operator Support


| Tengine              | Caffe                | MXNet                 | TensorFlow            | TF-Lite                      | ONNX              |
| -------------------- | -------------------- | --------------------- | --------------------- | ---------------------------- | ----------------- |
| Accuracy             | √                    |                       |                       |                              |                   |
| BatchNormalization   | BatchNorm            | BatchNorm             | FusedBatchNorm        |                              | √                 |
|                      |                      |                       | ComposedBN            |                              |                   |
| Resize               |                      |                       |                       | RESIZE_NEAREST_NEIGHBOR      |                   |
| Concat               | √                    | √                     | ConcatV2              | CONCATENATION                | √                 |
| Const                |                      |                       |                       |                              |                   |
| Convolution          | √                    | √                     | Conv2D                | CONV_2D                      | Conv              |
|                      | DepthwiseConvolution |                       | DepthwiseConv2dNative | DEPTHWISE_CONV_2D            |                   |
|                      | ConvolutionDepthwise |                       |                       |                              |                   |
| Deconvolution        | √                    | √                     | Conv2DBackpropInput   |                              |                   |
| DetectionOutput      | √                    |                       |                       |                              |                   |
| Dropout              | √                    | Copy                  | √                     |                              | √                 |
| Eltwise              | √                    | _minus_scalar         | Add                   | ADD                          | Add               |
|                      |                      | _mul_scalar           | Sub                   | SUB                          | Sub               |
|                      |                      | elemwise_add          |                       | PROD                         |                   |
|                      |                      |                       | Rsqrt                 | RSQRT                        |                   |
|                      |                      | _div_scalar           | RealDiv               | DIV                          | Div               |
|                      |                      |                       | Log                   | LOG                          |                   |
|                      |                      |                       | Exp                   | EXP                          | Exp               |
|                      |                      |                       | Pow                   | POW                          |                   |
|                      |                      |                       | Sqrt                  | SQRT                         |                   |
|                      |                      |                       | Floor                 | FLOOR                        | Floor             |
|                      |                      |                       | Mul                   | MUL                          | Mul               |
|                      |                      |                       | Minimum               |                              |                   |
|                      |                      |                       | AddN                  |                              |                   |
| Flatten              | √                    | √                     | √                     |                              | √                 |
| FullyConnected       | InnerProduct         | √                     | MatMul                | FULLY_CONNECTED              | Gemm              |
| Input                | Data                 |                       | FIFOQueueV2           |                              |                   |
|                      | Input                |                       |                       |                              |                   |
| LRN                  | √                    |                       | √                     |                              |                   |
| Normalize            | √                    |                       |                       |                              |                   |
| Permute              | √                    | transpose             |                       |                              |                   |
| Pooling              | √                    | √                     | AvgPool               | AVERAGE_POOL_2D              | AveragePool       |
|                      |                      |                       |                       |                              | GlobalAveragePool |
|                      |                      |                       | MaxPool               | MAX_POOL_2D                  | MaxPool           |
| PReLU                | √                    | LeakyReLU             |                       |                              | PRelu             |
| PriorBox             | √                    |                       |                       |                              |                   |
| Region               | √                    |                       |                       |                              |                   |
| ReLu                 | √                    | Activation            | Relu                  |                              | Relu              |
|                      |                      | LeakyReLU             |                       |                              | LeakyRelu         |
| ReLu6                | √                    | clip                  | Relu6                 |                              |                   |
| Reorg                | √                    |                       |                       |                              |                   |
| Reshape              | √                    | √                     | √                     | RESHAPE                      | √                 |
| ROIPooling           | √                    |                       |                       |                              |                   |
| RPN                  | √                    |                       |                       |                              |                   |
| Scale                | √                    |                       |                       |                              |                   |
| Slice                | √                    |                       |                       |                              | √                 |
| Softmax              | √                    | Activation            | √                     | SOFTMAX                      | √                 |
|                      | SoftmaxWithLoss      |                       |                       |                              |                   |
|                      |                      | SoftmaxOutput         |                       |                              |                   |
|                      |                      | SoftmaxActivation     |                       |                              |                   |
| Split                | √                    |                       | √                     |                              | √                 |
| DetectionPostProcess |                      |                       |                       | TFLite_Detection_PostProcess |                   |
| Gemm                 |                      |                       |                       |                              |                   |
|                      |                      |                       |                       |                              |                   |
| Generic              |                      |                       | DecodeWav             |                              |                   |
|                      |                      |                       | AudioSpectrogram      |                              |                   |
|                      |                      |                       | Mfcc                  |                              |                   |
| Logistic             |                      |                       |                       | LOGISTIC                     |                   |
| LSTM                 |                      | RNN                   | √                     |                              |                   |
| RNN                  |                      |                       | √                     |                              |                   |
| Tanh                 | TanH                 | Activation            | √                     |                              | √                 |
| Sigmoid              | √                    | Activation            | √                     |                              | √                 |
| Squeeze              |                      |                       |                       | SQUEEZE                      |                   |
| Pad                  |                      |                       | √                     |                              |                   |
|                      |                      |                       | MirrorPad             |                              |                   |
| StridedSlice         |                      |                       | √                     | STRIDED_SLICE                |                   |
| Reduction            | √                    | √                     | Sum                   | SUM                          |                   |
|                      |                      |                       | Mean                  | MEAN                         |                   |
|                      |                      |                       | Asum                  |                              |                   |
|                      |                      |                       | Sqsum                 |                              |                   |
|                      |                      |                       | Max                   |                              |                   |
|                      |                      |                       | Min                   |                              |                   |
|                      |                      |                       | Prod                  |                              |                   |
|                      |                      |                       | L2                    |                              |                   |
|                      |                      |                       | Logsum                |                              |                   |
|                      |                      |                       | Logsumexp             |                              |                   |
| ArgMax               |                      |                       | √                     |                              |                   |
| ArgMin               |                      |                       | √                     |                              |                   |
| TopKV2               |                      |                       | √                     |                              |                   |
| Maximum              |                      |                       | √                     |                              |                   |
| Minimum              |                      |                       | √                     |                              |                   |
| Addn                 |                      | add_n                 |                       |                              |                   |
| SwapAxis             |                      | √                     |                       |                              |                   |
| GRU                  |                      | RNN                   | √                     |                              |                   |
| Upsample             | √                    | UpSampling            |                       |                              |                   |
| ShuffleChannel       | √                    |                       |                       |                              |                   |
| Resize               | √                    |                       | ResizeNearestNeighbor |                              |                   |
|                      |                      |                       | ResizeBilinear        |                              |                   |
| SpaceToBatchND       |                      |                       | √                     |                              |                   |
| BatchToSpaceND       |                      |                       | √                     |                              |                   |
| Crop                 | √                    | √                     |                       |                              |                   |
| Psroipooling         |                      | _contrib_PSROIPooling |                       |                              |                   |
| Roialign             |                      | _contrib_ROIAlign     |                       |                              |                   |
| Expanddims           |                      |                       | ExpandDims            |                              |                   |
| Unary                |                      |                       | √                     |                              |                   |
|                      |                      | abs                   | Abs                   |                              |                   |
|                      |                      | neg                   | Neg                   |                              |                   |
|                      |                      | ceil                  | Ceil                  |                              |                   |
|                      |                      | floor                 | Floor                 |                              |                   |
|                      |                      | sin                   | Sin                   |                              |                   |
|                      |                      |                       | Asin                  |                              |                   |
|                      |                      | cos                   | Cos                   |                              |                   |
|                      |                      |                       | Acos                  |                              |                   |
|                      |                      | atan                  | Atan                  |                              |                   |
|                      |                      | tan                   | Tan                   |                              |                   |
|                      |                      |                       |                       |                              |                   |
|                      |                      | reciprocal            | Reciprocal            |                              |                   |
|                      |                      |                       | Square                |                              |                   |
|                      |                      |                       | Sqrt                  |                              |                   |
|                      |                      |                       | Rsqrt                 |                              |                   |
|                      |                      |                       | Exp                   |                              |                   |
|                      |                      |                       | Log                   |                              |                   |
| Bias                 | √                    |                       |                       |                              |                   |
| Noop                 |                      |                       |                       |                              |                   |
| Threshold            | √                    |                       |                       |                              |                   |
| Hardsigmoid          |                      |                       |                       |                              |                   |
| Embedding            | √                    | √                     | √                     |                              |                   |
| InstanceNorm         |                      | √                     |                       |                              |                   |
| MVN                  | √                    |                       |                       |                              |                   |
| Absval               | √                    |                       |                       |                              |                   |
| Cast                 |                      |                       | √                     |                              |                   |
| HardSwish            |                      |                       |                       |                              | √                 |
| Interp               | √                    | UpSampling            |                       |                              | Upsample          |
| Selu                 |                      |                       |                       |                              |                   |
| Elu                  | √                    | LeakyReLU             |                       | ELU                          | √                 |
| BroadMul             |                      | broadcast_mul         |                       |                              |                   |
| Logical              |                      |                       |                       | LOGICALOR                    |                   |
|                      |                      |                       |                       | LOGICALAND                   |                   |
| Gather               |                      |                       |                       | GATHER                       |                   |
| Transpose            |                      |                       | √                     | TRANSPOSE                    | √                 |
| Comparison           |                      |                       | Equal                 | EQUAL                        |                   |
|                      |                      |                       | Greater               | GREATER                      |                   |
|                      |                      |                       | GreaterEqual          | GREATER_EQUAL                |                   |
|                      |                      |                       | Less                  | LESS                         |                   |
|                      |                      |                       | LessEqual             |                              |                   |
|                      |                      |                       |                       | LESS_GREATER                 |                   |
| SpaceToDepth         |                      |                       |                       | SPACE_TO_DEPTH               |                   |
| DepthToSpace         |                      |                       |                       | DEPTH_TO_SPACE               |                   |
| Reverse              |                      |                       | ReverseV2             | REVERSE_V2                   |                   |
| SparseToDense        |                      |                       | √                     | SPARSE_TO_DENSE              |                   |
| Ceil                 |                      |                       | √                     | CEIL                         |                   |
| SquaredDifference    |                      |                       | √                     | SQUARED_DIFFERENCE           |                   |
| Round                |                      |                       | √                     | ROUND                        |                   |
| ZerosLike            |                      |                       |                       |                              |                   |
| Clip                 | Clip                 |                       |                       |                              | Clip              |
| Power                | Power                |                       |                       |                              |                   |
| Tile                 | Tile                 |                       |                       |                              |                   |
| L2Normalization      |                      |                       |                       | L2_NORMALIZATION             |                   |
| L2Pool               |                      |                       |                       | L2_POOL_2D                   |                   |
| Relu1                |                      |                       |                       | RELU_N1_TO_1                 |                   |
| LogSoftmax           |                      |                       |                       | LOG_SOFTMAX                  |                   |
| Floor                |                      |                       | Floor                 |                              |                   |
