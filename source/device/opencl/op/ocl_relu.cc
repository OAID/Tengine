

#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "relu_param.h"
}



bool OCLEngine::AddReluNode(struct node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_RELU(%d).\n",ir_node->index);

    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct relu_param* param = (struct relu_param*)ir_node->op.param_mem;


    //TLOG_INFO("Log cl kernel path: %s\n",cl_kernel_path);
    if (param->negative_slope > 0.000001)
    {
        char* cl_env = getenv("ROOT_PATH");
        char cl_kernel_path[500] = "";
        strcat(cl_kernel_path, cl_env);
        strcat(cl_kernel_path, "/source/device/opencl/cl/leaky_relu.cl");
        this->build_kernel(&cl_kernel_path[0], "leaky_relu");
    }
    else
    {
        char* cl_env = getenv("ROOT_PATH");
        char cl_kernel_path[500] = "";
        strcat(cl_kernel_path, cl_env);
        strcat(cl_kernel_path, "/source/device/opencl/cl/relu.cl");
        this->build_kernel(&cl_kernel_path[0], "relu");
    }

    int arg_idx = 0;
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

    struct OCLqueue Relu;
    Relu.name = "Relu";
    Relu.dims = 1;
    Relu.queue_kernel = this->kernel;
    Relu.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
    Relu.queue_global_work_size[0] = output_tensor->elem_num;
    Relu.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
    Relu.queue_local_work_size[0] =  1;
    this->queue_list.push_back(Relu);

    return true;
}


