
#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "slice_param.h"
}

bool OCLEngine::AddSliceNode(struct node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_FLATTEN(%d).\n", ir_node->index);

    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    char* cl_env = getenv("ROOT_PATH");
    char cl_kernel_path[500] = "";
    strcat(cl_kernel_path, cl_env);
    strcat(cl_kernel_path, "/source/device/opencl/cl/slice.cl");
//    TLOG_INFO( "Log cl kernel path: %s\n", cl_kernel_path);
    this->build_kernel(&cl_kernel_path[0], "slice");

    struct slice_param* param = (struct slice_param*)ir_node->op.param_mem;
    int res = 1;
    for (uint8_t i = input_tensor->dim_num-1; i > param->axis; i--)
    {
        res *= input_tensor->dims[i];
    }
    res *= param->begin;

    int arg_idx = 0;
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index])  );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &res) );

    struct OCLqueue Slice;
    Slice.name = "Slice";
    Slice.dims = 1;
    Slice.queue_kernel = this->kernel;
    Slice.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
    Slice.queue_global_work_size[0] = output_tensor->elem_num;
    Slice.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
    Slice.queue_local_work_size[0] =  1;
    this->queue_list.push_back(Slice);


    return true;

}


