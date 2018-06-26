# Operator Develop Guide

This document explains how to add a new operator into Tengine. 
- You should first add your **operator schema** in [operator](../operator) 
- then add your **operator implementation for execution** in [executor](../executor/operator).
    * executor/operator/arm64 (use AArch64 Instruction Set)
    * executor/operator/common (pure C++/ use openblas)


## Operator Schema
The operator schema will provide those information:
- `Operator name`: the name represents the operation will be taken on the input tensors.
- `Input/Output tensor` description: the place holder name, the data type, data layout(optional).
-` Operator parameter` definition and the default values, if any.
- `Document` to describe what the operator will do and requirements on input/output.
- `InferShape` to calculate the shape of output tensors according the shape of input tensors.


### 1. Operator with no Parameter
The operator must be derived from:
```c++
template <typename T>  OperatorNoParam
```
    
and implements:

- Operator(): default constructor to set the operator name
- Operator(const Operator&): copy constructor, which will be used by clone() interface
- SetSchema(): to set the input/output description, and the default values of parameters
- InferShape(): to calculate the output tensor shape. It is optional, as it is needed only when the output shapes are different with the input tensor shape.

Here is an example of `ReLu` Operator:
```c++
class ReLu: public OperatorNoParam<ReLu> {

public:

      ReLu() { name_="ReLu";}
      ReLu(const ReLu& src)=default;
      virtual ~ReLu() {};

      float GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs);

      void SetSchema(void) override;

};

```
Please refer to: [operator/include/operator/relu.hpp](../operator/include/operator/relu.hpp) and [operator/operator/relu.cpp](../operator/operator/relu.cpp) for more information.

### 2. Operator with Parameter
First, a separate parameter definition file should be created. In order to facilitate the parameter parsing, it is suggested to define the parameter structure following the example below:
```c++
struct ConvParam : public NamedParam {

   int  kernel_h;
   int  kernel_w;
   int  stride_h;
   int  stride_w;
   int  pad_h;
   int  pad_w;
   int  dilation_h;
   int  dilation_w;
   int  output_channel;
   int  group;


   DECLARE_PARSER_STRUCTURE(ConvParam) {
       DECLARE_PARSER_ENTRY(kernel_h);
       DECLARE_PARSER_ENTRY(kernel_w);
       DECLARE_PARSER_ENTRY(stride_h);
       DECLARE_PARSER_ENTRY(stride_w);
       DECLARE_PARSER_ENTRY(pad_h);
       DECLARE_PARSER_ENTRY(pad_w);
       DECLARE_PARSER_ENTRY(dilation_h);
       DECLARE_PARSER_ENTRY(dilation_w);
       DECLARE_PARSER_ENTRY(output_channel);
       DECLARE_PARSER_ENTRY(group);
   };

};
```
Then, the operator MUST be derived from:
```c++
template <typename T, typename P> OperatorWithParam
```
Only one additional interface, ParseParam(), may need to be implemented, just in case the parameter parsing cannot be handled easily with pre-defined methods.

Here is an example of Convolution operator definition.
```c++
class Convolution: public  OperatorWithParam <Convolution,ConvParam> {

public:
      Convolution(void) { name_="Convolution"; }
      Convolution(const Convolution&) =default;

      void SetSchema(void) override;

      bool InferShape(const std::vector<TEngine::TShape>&, std::vector<TEngine::TShape>&) override;
      float GetFops(const std::vector<TEngine::TShape>&, const std::vector<TEngine::TShape>&) override;

};
```

Please refer to: [operator/include/operator/conv_param.hpp](../operator/include/operator/conv_param.hpp) and [operator/include/operator/convolution.hpp](../operator/include/operator/convolution.hpp).

### 3. Register Operator

The new operator must register itself into system, so that other modules can create the operator.

This helper function must be called in some where:
```c++
template <typename T>
void RegisterOp(const std::string& name)
```

Please refers to the [operator/plugin/init.cpp](../operator/plugin/init.cpp): 


## An Example of Scale Operator
In the following, we will use `Scale` operator as our example to show how to add it into Tengine step by step. We will list related file names in the following steps.

### **Step1: add operator header file**
You need to create a header file. If have parameter, add param header file.
* [operator/include/operator/scale.hpp](../operator/include/operator/scale.hpp)
* [operator/include/operator/scale_param.hpp](../operator/include/operator/scale_param.hpp)

You define `Scale` class:
```c++
class Scale: public OperatorWithParam<Scale,ScaleParam> {
public:

    Scale() { name_="Scale"; }
    Scale(const Scale&)= default;
    ~Scale() {}

    void SetSchema(void) override;
};
```
and `ScaleParam`:
```c++
struct ScaleParam {
   int   axis;
   int   num_axes;
   int   bias_term;

   DECLARE_PARSER_STRUCTURE(ScaleParam) {
       DECLARE_PARSER_ENTRY(axis);
       DECLARE_PARSER_ENTRY(num_axes);
       DECLARE_PARSER_ENTRY(bias_term);
   };
};
```

### **Step2: add operator cpp file**

* [operator/operator/scale.cpp](../operator/operator/scale.cpp)
* [operator/operator/Makefile](../operator/operator/Makefile)

Set Scale operator schema, including Input, Output, etc.
```c++
void Scale::SetSchema(void)
{
   Input({"input:float32","gamma:float32","bias:float32"})
   .Output({"output:float32"})
   .SetAttr("axis",1)
   .SetAttr("num_axes",1)
   .SetAttr("bias_term",0)
   .SetDoc(R"DOC(Scale: only caffe flavor scale)DOC");
}
```
Remember to add `obj-y+=scale.o` in Makefile

### **Step3: register operator**
* [operator/plugin/init.cpp](../operator/plugin/init.cpp)

Add 
```c++
 RegisterOp<Scale>("Scale");
```
in plugin initial file to register Scale operator.

### **Step4: add implementation in executor**

* [executor/operator/common/scale.cpp](../executor/operator/common/scale.cpp)
* [executor/operator/common/Makefile](../executor/operator/common/Makefile)

The implementation is usually under the `Run` function and then `RegisterScaleNodeExec`:
```c++
namespace ScaleImpl 
{
   struct ScaleOps: public NodeOps 
   {
        bool Run(Node * node)
        {
            // your implementation
        }
    };
 }
 
using namespace ScaleImpl;
void RegisterScaleNodeExec(void)
{
    ScaleOps * ops=new ScaleOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common",
                "Scale",ops);
}
```
remember to add `obj-y+=scale.o` in Makefile

### **Step5: register Implementation in executor**
* [executor/plugin/init.cpp](../executor/plugin/init.cpp)

Add 
```c++
    extern void RegisterScale_NodeExec(void);
    RegisterScale_NodeExec();
```
in plugin initial file.

### **Step6: test operator implementation**
If you want to test your operator implementation, you can add test file. Remember to add `test_scale.o` in Makefile.
* [executor/tests/test_scale.cpp](../executor/tests/test_scale.cpp)
* [executor/tests/Makefile](../executor/tests/Makefile)


## Dynamic shape for Operator
Tengine also support dynamic shape for operator. Examples are `RPN` (in faster_rcnn) and `detection_output`(in SSD). The following will explain how to implement this method.

1. you should tell the network from which operator, the shape will be computed dynamically. You should add 
    ```
        SetOperatorDynamicShape(op);
    ```
    in your loadcaffe function, for example, in `LoadCaffeDetectionOutput()` and `LoadCaffeRPN()` in [serializer/caffe/caffe_serializer.cpp](../serializer/caffe/caffe_serializer.cpp) .

2. add DynamicProcess and do infer-shape in Run function, ref see [executor/operator/common/rpn.cpp](../executor/operator/common/rpn.cpp)
 ```
    bool Run(Node *node)
    {


        // compute num_box

        Tensor *output_tensor = node->GetOutputTensor(0);
        TShape &out_shape = output_tensor->GetShape();
        std::vector<int> outdim={1,num_box,4,1};
        out_shape.SetDim(outdim);

        // allocate out_mem
    }
```