

#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "pooling_param.h"
}



bool OCLEngine::AddPoolingNode(struct node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_POOL(%d).\n",ir_node->index);

    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct pool_param* params = ( struct pool_param* )ir_node->op.param_mem;

    char* cl_env = getenv("ROOT_PATH");
    char cl_kernel_path[500] = "";
    strcat(cl_kernel_path, cl_env);
    strcat(cl_kernel_path, "/source/device/opencl/cl/pool.cl");
    //TLOG_INFO("Log cl kernel path: %s\n",cl_kernel_path);
    if (params->global == 1)
    {
        this->build_kernel(&cl_kernel_path[0], "global_pool_avg");
    }
    else if (params->pool_method == 0)
    {
        this->build_kernel(&cl_kernel_path[0], "pool_max");
    }
    else if (params->pool_method == 1)
    {
        this->build_kernel(&cl_kernel_path[0], "pool_avg");
    }

    if (params->kernel_h == input_tensor->dims[2] && params->kernel_w == input_tensor->dims[3])
        params->global = 1;

    if (params->global)
    {
        params->pad_h0 = 0;
        params->pad_h1 = 0;
        params->pad_w0 = 0;
        params->pad_w1 = 0;
        params->kernel_h = input_tensor->dims[2];
        params->kernel_w = input_tensor->dims[3];
        params->pad_h0 = params->pad_h1 = params->pad_w0 = params->pad_w1 = 0;
        params->stride_h = params->stride_w = 1;
    }

    int arg_idx = 0;
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(int), &output_tensor->elem_num) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &input_tensor->dims[1]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &input_tensor->dims[2]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &input_tensor->dims[3]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->dims[2]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->dims[3]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &params->kernel_h) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &params->kernel_w) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &params->stride_h) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &params->stride_w) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &params->pad_h0) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &params->pad_w0) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]) );

    struct OCLqueue Pooling;
    Pooling.name = "Pooling";
    Pooling.dims = 1;
    Pooling.queue_kernel = this->kernel;
    Pooling.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
    Pooling.queue_global_work_size[0] = output_tensor->elem_num;
    Pooling.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
    Pooling.queue_local_work_size[0] =  1;
    this->queue_list.push_back(Pooling);





    return true;

}


