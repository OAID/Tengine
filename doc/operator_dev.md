# Operator Develop Guide

This document explains how to add a new operator into Tengine. 

To add a new operator, you should:

1. Register the new operator in the directory [Tengine/operator/](../operator). 

    Operator schema defines an interface for the operator's functionality, which is independent of the operator's implemenation. The operator schema defines:
    * op name
    * op inputs and outputs
    * op parameters
    * infershape function: used for tensor shape inference in `Prerun` stage.


2. Implement the operator in the directory [Tengine/executor/](../executor/operator). 
    
    This is the concrete implementation for operator. There can be multiple implementations for different parameters (for example, conv1x1, conv3x3) or architectures (for example, armv8/armv7). 


## An Example of Scale Operator
In this section, we use `Scale` operator as our example to show how to add it in Tengine step by step.

### Step 1: Create Operator Header File
There are two type of operators:
- Operator without parameters: `template <typename T>  OperatorNoParam`
- Operator with parameters: `template <typename T, typename P> OperatorWithParam`

The scale operator is an operator with parameters.

* Create `ScaleParam` file [operator/include/operator/scale_param.hpp](../operator/include/operator/scale_param.hpp):
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
* Create `Scale` file [operator/include/operator/scale.hpp](../operator/include/operator/scale.hpp):
    ```c++
    class Scale: public OperatorWithParam<Scale,ScaleParam> {
    public:

        Scale() { name_="Scale"; }
        Scale(const Scale&)= default;
        ~Scale() {}

        void SetSchema(void) override;
    };
    ```
### **Step 2: Create Operator Cpp File**

* Set operator schema in file [operator/operator/scale.cpp](../operator/operator/scale.cpp):
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

* Add `obj-y+=scale.o` in file [operator/operator/Makefile](../operator/operator/Makefile).



### **Step 3: Register Operator in Operator Plugin**
* Add register scale operator in file [operator/plugin/init.cpp](../operator/plugin/init.cpp):
    ```c++
    RegisterOp<Scale>("Scale");
    ```

### **Step 4: Create Implementation in Executor Folder**

* Add the implementation of scale operator in file [executor/operator/common/scale.cpp](../executor/operator/common/scale.cpp) and register the implemetation in the function `RegisterScaleNodeExec`:

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
* Add `obj-y+=scale.o` in file [executor/operator/common/Makefile](../executor/operator/common/Makefile).

### **Step 5: Register Implementation in Executor Plugin**

* Add `RegisterScale_NodeExec` in file [executor/plugin/init.cpp](../executor/plugin/init.cpp):
    ```c++
        extern void RegisterScale_NodeExec(void);
        ...
        RegisterScale_NodeExec();
    ```

