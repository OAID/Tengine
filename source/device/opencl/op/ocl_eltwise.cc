

#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "eltwise_param.h"
}


bool OCLEngine::AddEltwiseNode(struct node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_ELTWISE(%d).\n",ir_node->index);

    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct eltwise_param* params = ( struct eltwise_param* )ir_node->op.param_mem;

    char* cl_env = getenv("ROOT_PATH");
    char cl_kernel_path[500] = "";
    strcat(cl_kernel_path, cl_env);
    strcat(cl_kernel_path, "/source/device/opencl/cl/eltwise.cl");
    //TLOG_INFO("Log cl kernel path: %s\n",cl_kernel_path);

    switch (params->type)
    {
        case ELT_SUM:
        {
            this->build_kernel(&cl_kernel_path[0], "eltwise_add");
            break;
        }
        default:
            break;
    }

    int arg_idx = 0;
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor0->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor1->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

    struct OCLqueue Eltwise;
    Eltwise.name = "Eltwise";
    Eltwise.dims = 1;
    Eltwise.queue_kernel = this->kernel;
    Eltwise.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
    Eltwise.queue_global_work_size[0] = output_tensor->elem_num;
    Eltwise.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
    Eltwise.queue_local_work_size[0] =  output_tensor->dims[1]/2;
    this->queue_list.push_back(Eltwise);

    return true;
}

