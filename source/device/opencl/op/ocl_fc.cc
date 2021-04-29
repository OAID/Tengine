
#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "convolution_param.h"
}


bool OCLEngine::AddFullyConnectionNode(struct node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_FC(%d).\n",ir_node->index);

    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* bias_tensor;
    if (2 < ir_node->input_num)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    cl_event enentPoint;
    CHECK_ENQUEUE_BUFFER_STATUS(clEnqueueWriteBuffer(commandQueue, this->ocl_tensor_map[weight_tensor->index], CL_TRUE, 0, weight_tensor->elem_num * weight_tensor->elem_size, weight_tensor->data, 0, NULL, &enentPoint));
    clWaitForEvents(1,&enentPoint); ///wait
    clReleaseEvent(enentPoint);

    if (2 < ir_node->input_num)
    {
        cl_event enentPoint;
        CHECK_ENQUEUE_BUFFER_STATUS(clEnqueueWriteBuffer(commandQueue, this->ocl_tensor_map[bias_tensor->index], CL_TRUE, 0, bias_tensor->elem_num * bias_tensor->elem_size, bias_tensor->data, 0, NULL, &enentPoint));
        clWaitForEvents(1,&enentPoint); ///wait
        clReleaseEvent(enentPoint);
    }

    char* cl_env = getenv("ROOT_PATH");
    char cl_kernel_path[500] = "";
    strcat(cl_kernel_path, cl_env);
    strcat(cl_kernel_path, "/source/device/opencl/cl/mat_mul0.cl");
    //TLOG_INFO("Log cl kernel path: %s\n",cl_kernel_path);
    this->build_kernel(&cl_kernel_path[0], "mat_mul");

    int M = output_tensor->dims[1];
    int N;
    if (output_tensor->dim_num > 2)
        N = output_tensor->dims[2] * output_tensor->dims[3];
    else
        N = 1;
    int K;
    if (weight_tensor->dim_num > 2)
        K = weight_tensor->dims[1] * weight_tensor->dims[2] * weight_tensor->dims[3];
    else
        K = weight_tensor->dims[1];

    int arg_idx = 0;
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[weight_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &M) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &N) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &K) );

    struct OCLqueue Mat_Mul;
    Mat_Mul.name = "Mat_Mul";
    Mat_Mul.dims = 1;
    Mat_Mul.queue_kernel = this->kernel;
    Mat_Mul.queue_global_work_size = (size_t*)malloc(1 * sizeof(size_t));
    Mat_Mul.queue_global_work_size[0] = M * N;
    Mat_Mul.queue_local_work_size = (size_t*)malloc(1 * sizeof(size_t));
    Mat_Mul.queue_local_work_size[0] = M/4;
    this->queue_list.push_back(Mat_Mul);

    /* bias_add and relu compute */
    if (2 < ir_node->input_num)
    {
        int elem_23 = output_tensor->dims[2] * output_tensor->dims[3];
        int elem_123 = output_tensor->dims[1] * elem_23;

        char cl_kernel_path2[500] = "";
        strcat(cl_kernel_path2, cl_env);
        strcat(cl_kernel_path2, "/source/device/opencl/cl/bias_add.cl");
        //TLOG_INFO("Log cl kernel path: %s\n",cl_kernel_path2);
        this->build_kernel(&cl_kernel_path2[0], "bias_add");

        arg_idx = 0;
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[bias_tensor->index]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_123) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_23) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

        struct OCLqueue Bias_Add;
        Bias_Add.name = "Bias_Add";
        Bias_Add.dims = 1;
        Bias_Add.queue_kernel = this->kernel;
        Bias_Add.queue_global_work_size = (size_t*)malloc(1 * sizeof(size_t));
        Bias_Add.queue_global_work_size[0] = M * N;
        Bias_Add.queue_local_work_size = (size_t*)malloc(1 * sizeof(size_t));
        Bias_Add.queue_local_work_size[0] = M/4;
        this->queue_list.push_back(Bias_Add);
    }


    return true;
}












