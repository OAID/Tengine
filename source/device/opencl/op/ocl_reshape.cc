
#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "operator/op.h"
}

bool OCLEngine::AddReshapeNode(struct node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_RESHAPE(%d).\n", ir_node->index);

    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    char* cl_env = getenv("ROOT_PATH");
    char cl_kernel_path[500] = "";
    strcat(cl_kernel_path, cl_env);
    strcat(cl_kernel_path, "/source/device/opencl/cl/reshape.cl");
//    TLOG_INFO( "Log cl kernel path: %s\n", cl_kernel_path);
    this->build_kernel(&cl_kernel_path[0], "reshape");

    int arg_idx = 0;
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

    struct OCLqueue Reshape;
    Reshape.name = "Reshape";
    Reshape.dims = 1;
    Reshape.queue_kernel = this->kernel;
    Reshape.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
    Reshape.queue_global_work_size[0] = output_tensor->elem_num;
    Reshape.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
    Reshape.queue_local_work_size[0] =  1;
    this->queue_list.push_back(Reshape);


    return true;

}


