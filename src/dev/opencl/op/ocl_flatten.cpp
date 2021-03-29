
#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "tengine_op.h"
}

bool OCLEngine::AddFlattenNode(struct ir_node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_FLATTEN(%d).\n", ir_node->idx);

    struct ir_graph* ir_graph = ir_node->graph;

    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    char* cl_env = getenv("ROOT_PATH");
    char cl_kernel_path[500] = "";
    strcat(cl_kernel_path, cl_env);
    strcat(cl_kernel_path, "/src/dev/opencl/cl/flatten.cl");
//    fprintf(stderr, "Log cl kernel path: %s\n", cl_kernel_path);
    this->build_kernel(&cl_kernel_path[0], "flatten");

    int arg_idx = 0;
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->idx])  );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->idx]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

    struct OCLqueue Flatten;
    Flatten.name = "Flatten";
    Flatten.dims = 1;
    Flatten.queue_kernel = this->kernel;
    Flatten.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
    Flatten.queue_global_work_size[0] = output_tensor->elem_num;
    Flatten.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
    Flatten.queue_local_work_size[0] =  1;
    this->queue_list.push_back(Flatten);




}


