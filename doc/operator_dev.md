# Operator Develop Guide

This document explains how to add a new operator into TEngine. 
- You should first add your **operator schema** in [operator](../operator) 
- then add your **operator implementation for execution** in [executor](../executor).


## Operator Schema
The operator schema will provide those information:
- Operator name.The name represents the operation will be taken on the input tensors.
- Input/Ouput tensor description: the place holder name, the data type, data layout(optional)
- Operator parameter defintion and the default values, if any.
- Document to describe what the operator will do and requirements on input/output
- InferShape to calculate the shape of output tensors according the shape of input tensors


### 1.Operator with no Parameter
The operator must be derived from:
```
template <typename T>  OperatorNoParam
```
    
and implements:

- OP(): default constructor to set the operator name
- OP(const OP&): copy constructor, which will be used by clone() interface
- SetSchema(): to set the input/output description, and the default values of parameters
- InferShape(): to calculate the output tensor shape. It is optional, as it needs only when the output shapes are different with the input tensor shape.

Here is an example of `ReLu` Operator:
```
class ReLu: public OperatorNoParam<ReLu> {

public:

      ReLu() { name_="ReLu";}
      ReLu(const ReLu& src)=default;
      virtual ~ReLu() {};

      void SetSchema(void) override;

};

```
Please refer to: [operator/include/operator/relu.hpp](../operator/include/operator/relu.hpp) and [operator/operator/relu.cpp](../operator/operator/relu.cpp) for more informations.

### 2.Operator with Parameter
First, a seperator parameter definiton file should be created. In order to faciliate the parameter parsing, it is suggested to define the parameter structure following the example below:
```
struct ConvParam {

   int  kernel_h;
   int  kernel_w;
   int  stride_h;
   int  stride_w;
   int  pad_h;
   int  pad_w;
   int  output_channel;
   int  group;

   DECLARE_PARSER_STRUCTURE(ConvParam) {
       DECLARE_PARSER_ENTRY(kernel_h);
       DECLARE_PARSER_ENTRY(kernel_w);
       DECLARE_PARSER_ENTRY(stride_h);
       DECLARE_PARSER_ENTRY(stride_w);
       DECLARE_PARSER_ENTRY(pad_h);
       DECLARE_PARSER_ENTRY(pad_w);
       DECLARE_PARSER_ENTRY(output_channel);
       DECLARE_PARSER_ENTRY(group);
   };

};
```
Then, the operator MUST be derived from:
```
template <typename T, typename P> OperatorWithParam
```
Only one additonal interface, ParseParam(), may need to be implement, just in case the parameter parsing cannot be handled easily with pre-defined methods.

Here is an example of Convolution operator definiton.
```
class Convolution: public  OperatorWithParam <Convolution,ConvParam> {

public:
      Convolution(void) { name_="Convolution"; }
      Convolution(const Convolution&) =default;

      void SetSchema(void) override;

      bool InferShape(const std::vector<TEngine::TShape>&, std::vector<TEngine::TShape>&) override;

};
```

Please refer to: [operator/include/operator/conv_param.hpp](../operator/include/operator/conv_param.hpp) and [operator/include/operator/convolution.hpp](../operator/include/operator/convolution.hpp).

### 3.Register Operator

The new operator must register itself into system, so that other modules can create the operator.

This helpper function must be called in some where:
```
	template <typename T>
	void RegisterOp(const std::string& name)
```

Please refers to the [operator/plugin/init.cpp](../operator/plugin/init.cpp): 



## An Example of Scale Operator
In the following, we will use `Scale` operator as our example to show how to  add it into Tengine step by step. We will list related file names in the following steps.

### Step1: add operator header file
You need to create a header file. If have parameter, add param header file.
* [operator/include/operator/scale.hpp](../operator/include/operator/scale.hpp)
* [operator/include/operator/scale_param.hpp](../operator/include/operator/scale_param.hpp)

You define `Scale` class:
```
class Scale: public OperatorWithParam<Scale,ScaleParam> {
public:

    Scale() { name_="Scale"; }
    Scale(const Scale&)= default;
    ~Scale() {}

    void SetSchema(void) override;
};
```
and `ScaleParam`:
```
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

### Step2: add operator cpp file

* [operator/operator/scale.cpp](../operator/operator/scale.cpp)
* [operator/operator/Makefile](../operator/operator/Makefile)

Set Scale operator schema,including Input, Output, etc.
```
void Scale::SetSchema(void)
{
   Input({"input:float32"})
   .Output({"output:float32"})
   .SetAttr("axis",1)
   .SetAttr("num_axes",1)
   .SetAttr("bias_term",0)
   .SetDoc(R"DOC(Scale: only caffe flavor scale)DOC");
}
```
Remember to add `obj-y+=scale.o` in Makefile

### Step3: register opetator
* [operator/plugin/init.cpp](../operator/plugin/init.cpp)
Add 
```
 RegisterOp<Scale>("Scale");
```
in plugin initial file to register Scale operator.

### Step4: add implement in  executor


* [executor/operator/arm/scale.cpp](../executor/operator/arm/scale.cpp)
* [executor/operator/arm/Makefile](../executor/operator/arm/Makefile)

The implementation is usually under the `Run` function and then `RegisterScaleNodeExec`:
```
namespace ScaleImpl 
{
   bool Run(Node * node, ExecEngine * engine)
   {
   // your implementation
   }
 }
 
 void RegisterScaleNodeExec(void)
{
    NodeExec scale_exec={ScaleImpl::OnBind,nullptr,ScaleImpl::Run,nullptr};

    RegisterNodeExec("Scale",scale_exec);
}
```
remember to add `obj-y+=scale.o` in Makefile

### Step5: register Implementation in executor
* [executor/plugin/init.cpp](../executor/plugin/init.cpp)

Add 
```
  RegisterScaleNodeExec();
```
in plugin initial file.

### Step6: test opetator implement
If you want to test your operator implement, you can add test file. Remember to  add `test_scale.o` in Makefile.
* [executor/tests/test_scale.cpp](../executor/tests/test_scale.cpp)
* [executor/tests/Makefile](../executor/tests/Makefile)
