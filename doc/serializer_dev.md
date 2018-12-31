# Tengine Serializer Develop Guide

## 1. Overview
This document defines the APIs and requirements to develop a serializer module for Tengine. 
Each serializer module works on one specific model format only, such Caffe/ONNX/Mxnet/Tensorflow/Tengine.
The serializer module loads the whole model file stored in disk, and creates a Tengine in-memory IR, which is StaticGraph. The serializer module also can store the StaticGraph into disk in the specific format. However, current version of this document describes the loading process, which is more important than the storing process.

The section 2, class serializer, introduces interfaces that the developer should implemented.
The Serializer module's load method **MUST** be MT-Safe: multiple threads can call the load method simultaneously. It is safe to assume that there is only **ONE** instance for a serializer module in system.

## 2. Class Serializer
All serializer module must be derived from class serializer. One serializer class should only implement one AI framework model load. While for a AI framework, it is possible that there are different serializer modules.

The interfaces of class serializer can be classified into below categories:

### 2.1 Load Interface
``` c++
unsigned int GetFileNum(void);
bool LoadModel(const std::vector<std::string>& file_list, StaticGraph * static_graph);
```
GetFileNum() will return the number of files that need to load a model.<br>
LoadModel() will do all the work to parse the saved model and convert it to StaticGraph. The static_graph is created by the caller in advance.

### 2.2 Information Interface
```c++
const std::string& GetFormatName(void);
const std::string& GetVersion(void);
const std::string& GetName(void);
```

### 2.3 Helper Function: Operator Load Function Registry
```c++
bool RegisterOpLoadMethod(const std::string& op_name,const any& load_func);
any& GetOpLoadMethod(const std::string& op_name);
```
Register operator load function for operator op_name. This enables developer to implement operator load function outside the serializer module.

### 2.4 Optional Helper Function
Const tensor reload helper:
```c++
bool LoadConstTensor(const std::string& fname, StaticTensor * const_tensor);
bool LoadConstTensor(int fd, StaticTensor * const_tensor);
```
The developer can use GetTensorName() to get the tensor name. The memory to hold the tensor data should be allocated by those functions and the memory owner will be transfered to the framework.

## 3. SerializerFactory and Serializer Object Manager Interface
The user of the serializer module will get a serializer object through SerializerManager. <br>
For example:
```c++
SerializerPtr p_onnx;
SerializerManager::SafeGet("onnx", p_onnx);
p_onnx->LoadModel(flist, graph);
```
Each serializer module should register an object into SerializerManager during initialization phase. As it is requested that the serializer load method is MT-Safe, only **ONE** serializer object is enough for whole system.

Here is an example work follow of creating and registering a serializer object. It is optional to use SerializerFactory to create a serializer object.
```c++
auto factory=SerializerFactory::GetFactory();
factory->RegisterInterface<OnnxSerializer>("onnx");

auto onnx_serializer=factory->Create("onnx");
SerializerManager::SafeAdd("onnx",SerializerPtr(onnx_serializer));
```

## 4. Static Graph API
The static graph is Tengine in-memory IR and it is opaque to serializer module developer. 
Four components are defined as the forward declaration and are manipulated through the static graph AI.
```c++
struct StaticGraph;
struct StaticNode;
struct StaticTensor;
struct StaticOp;
```

The description below only covers the loading process. 

As different operator has its own parameter definition, the serializer developer must include the header file `operator/<op_A>_param.hpp` and use the parameter defined there. After the parameters are loaded correctly, `SetOperatorParam()` should be called. As some parameters may be not recorded in the saved model file, it is required to get the default parameter first, by calling `OpManager::GetOpDefParam()`.

Here is an example of loading caffe's concat operator:
```c++
static bool LoadCaffeConcat(StaticGraph* graph, StaticNode* node, const te_caffe::LayerParameter& layer_param)
{
    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));
    const te_caffe::ConcatParameter& concat_param = layer_param.concat_param();

    if(concat_param.has_concat_dim())
        param.axis = static_cast<int>(concat_param.concat_dim());
    else
        param.axis = concat_param.axis();

    StaticOp* op = CreateStaticOp(graph, "Concat");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
```

The sections followed will list the API in detail. Please note that the serializer module **MUST not** release the object pointer, returned by CreateXXX API.

### 4.1 StaticGraph API
```c++
void DumpStaticGraph(StaticGraph * graph);
bool CheckGraphIntegraity (StaticGraph * graph);
void SetGraphInternalName(StaticGraph * graph, const std::string& name);
void SetGraphIdentity(StaticGraph * graph, const std::string& domain, const std::string& name, const std::string& version);
void SetGraphSource(StaticGraph * graph, const std::string& source);
void SetGraphSourceFormat(StaticGraph * graph, const std::string& format);
void SetGraphConstTensorFile(StaticGraph * graph, const std::string& fname);
bool AddGraphAttr(StaticGraph * graph, const std::string& attr_name, any&& value);
```

### 4.2 StaticNode API
```c++
StaticNode * CreateStaticNode(StaticGraph * graph, const std::string& node_name);
void SetNodeOp(StaticNode * node, StaticOp * op);
int AddNodeInputTensor(StaticNode * node, StaticTensor * tensor);
int AddNodeOutputTensor(StaticNode * node, StaticTensor * tensor);
const std::string& GetNodeName(StaticNode * node);
```

### 4.3 StaticOp API
```c++
StaticOp * CreateStaticOp(StaticGraph * graph, const std::string& op_name);
void SetOperatorParam(StaticOp * op, any&& param);
void AddOperatorAttr(StaticOp * op, const std::string& attr_name, any&& val);
```

### 4.4 StaticTensor API
```c++
StaticTensor * CreateStaticTensor(StaticGraph * graph, const std::string& name);
StaticTensor * CreateStaticConstTensor(StaticGraph * graph, const std::string& name);
void  SetTensorDim(StaticTensor * tensor, const std::vector<int>& dims);
void  SetTensorDataType(StaticTensor * tensor, const std::string& data_type);
void  SetTensorDataLayout(StaticTensor * tensor, const std::string& data_layout);
void  SetTensorType(StaticTensor * tensor, int type); 
int   SetTensorSize(StaticTensor * tensor, int size);
void  SetConstTensorBuffer(StaticTensor * tensor, void * addr);
void  SetConstTensorFileLocation(StaticTensor * tensor, int offset, int file_size);
const std::string& GetTensorName(StaticTensor * tensor);
```

## 5. Major Header files

Here are the major external header files necessary to develop a serializer module.

    serializer.hpp  :  the Serializer Interface as well as the SerializerManager
    operator_manager.hpp  :  OpManager::GetOpDefParam
    static_graph_interface.hpp  :  static graph interface
    operator/xxx_param.hpp  :  Operator parameter

## 6. Example code

Refers to Tengine code repository: [serializer/caffe/caffe_serializer.cpp](../serializer/caffe/caffe_serializer.cpp) and [serializer/onnx/onnx_serializer.cpp](../serializer/onnx/onnx_serializer.cpp)
