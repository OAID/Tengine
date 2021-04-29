

#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "concat_param.h"
}


bool OCLEngine::AddConcatNode(struct node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_CONCAT(%d).\n",ir_node->index);

    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    if (ir_node->input_num == 1)
    {
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

        char* cl_env = getenv("ROOT_PATH");
        char cl_kernel_path[500] = "";
        strcat(cl_kernel_path, cl_env);
        strcat(cl_kernel_path, "/source/device/opencl/cl/reshape.cl");
//    TLOG_INFO( "Log cl kernel path: %s\n", cl_kernel_path);
        this->build_kernel(&cl_kernel_path[0], "reshape");

        int arg_idx = 0;
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor0->index]) );
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
    }
    else if (ir_node->input_num == 2)
    {
        struct tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

        struct concat_param* params = ( struct concat_param* )ir_node->op.param_mem;

        char* cl_env = getenv("ROOT_PATH");
        char cl_kernel_path[500] = "";
        strcat(cl_kernel_path, cl_env);
        strcat(cl_kernel_path, "/source/device/opencl/cl/concat.cl");
        this->build_kernel(&cl_kernel_path[0], "concat2");

        int axis_size;
        int pre_size = 1;
        int post_size = 1;
        for (int i = input_tensor0->dim_num-1; i >= params->axis; i--)
        {
            pre_size *= input_tensor0->dims[i];
        }
        for (int i = input_tensor1->dim_num-1; i >= params->axis; i--)
        {
            post_size *= input_tensor1->dims[i];
        }
        axis_size = pre_size + post_size;

        int arg_idx = 0;
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor0->index]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor1->index]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &axis_size) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &pre_size) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &post_size) );

        struct OCLqueue Concat;
        Concat.name = "Concat";
        Concat.dims = 1;
        Concat.queue_kernel = this->kernel;
        Concat.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
        Concat.queue_global_work_size[0] = output_tensor->elem_num;
        Concat.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
        Concat.queue_local_work_size[0] =  output_tensor->dims[1]/2;
        this->queue_list.push_back(Concat);
    }
    else
        TLOG_INFO("Log : Not Support up till now concat%d!",ir_node->input_num);


    return true;
}
