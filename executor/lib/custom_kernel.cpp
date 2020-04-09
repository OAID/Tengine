#include "data_type.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tengine_errno.hpp"
#include "custom_kernel.hpp"
#include "custom_kernel_ops.hpp"

namespace TEngine {

namespace CustomKernelManager {

static void PrepareOneTensor(Node* node, Tensor* tensor, struct custom_kernel_tensor* t)
{
    const TShape& shape = tensor->GetShape();
    const std::vector<int>& dims = shape.GetDim();

    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

    t->dim_num = dims.size();

    for(int i = 0; i < t->dim_num; i++)
    {
        t->dim[i] = dims[i];
    }

    t->data_type = tensor->GetDataType();
    t->element_num = shape.GetSize();
    t->element_size = DataType::GetTypeSize(t->data_type);
    t->layout_type = exec_attr->graph_layout;

    t->data = nullptr;
}

bool PrepareTensors(Node* node, struct custom_kernel_tensor**& k_inputs, int input_num,
                    struct custom_kernel_tensor**& k_outputs, int output_num)
{
    k_inputs = nullptr;
    k_outputs = nullptr;

    /*prepare inputs and outputs */
    if(input_num)
    {
        k_inputs = ( struct custom_kernel_tensor** )malloc(sizeof(void*) * input_num);

        if(k_inputs == nullptr)
        {
            set_tengine_errno(ENOMEM);
            return false;
        }

        memset(k_inputs, 0x0, sizeof(void*) * input_num);
    }

    if(output_num)
    {
        k_outputs = ( struct custom_kernel_tensor** )malloc(sizeof(void*) * output_num);

        if(k_outputs == nullptr)
        {
            set_tengine_errno(ENOMEM);
            return false;
        }

        memset(k_outputs, 0x0, sizeof(void*) * output_num);
    }

    for(int i = 0; i < input_num; i++)
    {
        struct custom_kernel_tensor* t = ( struct custom_kernel_tensor* )malloc(sizeof(struct custom_kernel_tensor));

        if(t == nullptr)
        {
            set_tengine_errno(ENOMEM);
            return false;
        }

        Tensor* tensor = node->GetInputTensor(i);

        PrepareOneTensor(node, tensor, t);

        k_inputs[i] = t;
    }

    for(int i = 0; i < output_num; i++)
    {
        struct custom_kernel_tensor* t = ( struct custom_kernel_tensor* )malloc(sizeof(struct custom_kernel_tensor));

        if(t == nullptr)
        {
            set_tengine_errno(ENOMEM);
            return false;
        }

        Tensor* tensor = node->GetOutputTensor(i);

        PrepareOneTensor(node, tensor, t);

        k_outputs[i] = t;
    }

    return true;
}

void ReleaseTensors(Node* node, struct custom_kernel_tensor** k_inputs, int input_num,
                    struct custom_kernel_tensor** k_outputs, int output_num)
{
    if(k_inputs != nullptr)
    {
        for(int i = 0; i < input_num; i++)
        {
            if(k_inputs[i] != nullptr)
                free(k_inputs[i]);
        }

        free(k_inputs);
    }

    if(k_outputs != nullptr)
    {
        for(int i = 0; i < output_num; i++)
        {
            if(k_outputs[i] != nullptr)
                free(k_outputs[i]);
        }

        free(k_outputs);
    }
}

static bool CheckBind(Node* node, struct custom_kernel_ops* ops)
{
    /* prepare tensors */
    int input_num = node->GetInputNum();
    int output_num = node->GetOutputNum();

    struct custom_kernel_tensor** k_inputs;
    struct custom_kernel_tensor** k_outputs;

    if(!PrepareTensors(node, k_inputs, input_num, k_outputs, output_num))
    {
        ReleaseTensors(node, k_inputs, input_num, k_outputs, output_num);
        return false;
    }

    int ret = ops->bind(ops, ( const struct custom_kernel_tensor** )k_inputs, input_num,
                        ( const struct custom_kernel_tensor** )k_outputs, output_num);

    ReleaseTensors(node, k_inputs, input_num, k_outputs, output_num);

    if(ret < 0)
        return false;

    return true;
}

NodeOps* BindOps(Node* node, const std::string& dev_name)
{
    CustomKernelList* k_list = any_cast<CustomKernelList>(&node->GetAttr(ATTR_CUSTOM_KERNEL));

    struct custom_kernel_ops* ops = k_list->GetKernel(dev_name.c_str());

    if(ops == nullptr)
    {
        /* try if there is any device support setting */
        ops = k_list->GetKernel(ANY_DEVICE_NAME);

        if(ops == nullptr)
            return nullptr;
    }

    Operator* op = node->GetOp();

    if(op->GetName() != "Generic" && (op->GetName() != ops->op))
    {
        XLOG_WARN() << "unmatched op: custom " << ops->op << " real " << op->GetName() << "\n";
        return nullptr;
    }

    /* check bind */
    if(ops->bind != nullptr && !CheckBind(node, ops))
    {
        if(ops->force)
        {
            LOG_ERROR() << "bind custom kernel for node: " << node->GetName() << " failed, but force is set\n";

            set_tengine_errno(ENOTRECOVERABLE);
        }

        return nullptr;
    }

    NodeOps* node_ops = CustomKernelNodeOps::NewOps(node, ops);

    return node_ops;
}

}    // namespace CustomKernelManager

}    // namespace TEngine
