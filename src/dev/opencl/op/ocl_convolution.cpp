
#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

extern "C"
{
#include "tengine_op.h"
#include "convolution_param.h"
}

void trans(float* data_in, float* data_out, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int in_idx = i * N + j;
            int out_idx = j * M + i;
            data_out[out_idx] = data_in[in_idx];
        }
    }
}

bool OCLEngine::AddConvolutionNode(struct ir_node* ir_node)
{
    TLOG_INFO("Tengine OpenCL: Support OP_CONV(%d).\n",ir_node->idx);

    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct ir_tensor* bias_tensor;
    if (2 < ir_node->input_num)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    float* data_trans = (float *)malloc(weight_tensor->elem_num * weight_tensor->elem_size);
    trans((float*)weight_tensor->data, data_trans, weight_tensor->dims[0], weight_tensor->elem_num / weight_tensor->dims[0]);
    cl_event enentPoint;
//    CHECK_ENQUEUE_BUFFER_STATUS(clEnqueueWriteBuffer(commandQueue, this->ocl_tensor_map[weight_tensor->idx], CL_TRUE, 0, weight_tensor->elem_num * weight_tensor->elem_size, data_trans, 0, NULL, &enentPoint));
    CHECK_ENQUEUE_BUFFER_STATUS(clEnqueueWriteBuffer(commandQueue, this->ocl_tensor_map[weight_tensor->idx], CL_TRUE, 0, weight_tensor->elem_num * weight_tensor->elem_size, weight_tensor->data, 0, NULL, &enentPoint));
    clWaitForEvents(1,&enentPoint); ///wait
    clReleaseEvent(enentPoint);

    if (2 < ir_node->input_num)
    {
        cl_event enentPoint;
        CHECK_ENQUEUE_BUFFER_STATUS(clEnqueueWriteBuffer(commandQueue, this->ocl_tensor_map[bias_tensor->idx], CL_TRUE, 0, bias_tensor->elem_num * bias_tensor->elem_size, bias_tensor->data, 0, NULL, &enentPoint));
        clWaitForEvents(1,&enentPoint); ///wait
        clReleaseEvent(enentPoint);
    }

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    if (conv_param->group != weight_tensor->dims[0]) // Direct Conv
    {
        char* cl_env = getenv("ROOT_PATH");
        char cl_kernel_path[500] = "";
        strcat(cl_kernel_path, cl_env);
        strcat(cl_kernel_path, "/src/dev/opencl/cl/im2col.cl");
        //fprintf(stderr,"Log cl kernel path: %s\n",cl_kernel_path);
        this->build_kernel(&cl_kernel_path[0], "im2col");

        int channels = input_tensor->dims[1];
        int height = input_tensor->dims[2];
        int width = input_tensor->dims[3];

        int height_col = (height + 2 * conv_param->pad_h0 - (conv_param->dilation_h * (conv_param->kernel_h - 1) + 1)) / conv_param->stride_h + 1;
        int width_col = (height + 2 * conv_param->pad_w0 - (conv_param->dilation_w * (conv_param->kernel_w - 1) + 1)) / conv_param->stride_w + 1;

        int col_chw = weight_tensor->dims[1] * weight_tensor->dims[2] * weight_tensor->dims[3] * height_col * width_col;

        int ir_tensor_idx = output_tensor->idx + ir_graph->tensor_num;
        cl_mem clBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE , col_chw * sizeof(float), NULL, NULL);
        this->ocl_tensor_map[ir_tensor_idx] = clBuffer;

        int arg_idx = 0;
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->idx]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &col_chw) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &height) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &width) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &weight_tensor->dims[1]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &weight_tensor->dims[2]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &weight_tensor->dims[3]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->pad_h0) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->pad_w0) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->stride_h) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->stride_w) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &height_col) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &width_col) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&clBuffer) );

        struct OCLqueue Im2Col;
        Im2Col.name = "Im2Col";
        Im2Col.dims = 1;
        Im2Col.queue_kernel = this->kernel;
        Im2Col.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
        Im2Col.queue_global_work_size[0] = col_chw;
        Im2Col.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
        Im2Col.queue_local_work_size[0] =  1;
        this->queue_list.push_back(Im2Col);

        char cl_kernel_path1[500] = "";
        strcat(cl_kernel_path1, cl_env);
        strcat(cl_kernel_path1, "/src/dev/opencl/cl/mat_mul.cl");
        //fprintf(stderr,"Log cl kernel path: %s\n",cl_kernel_path1);
        this->build_kernel(&cl_kernel_path1[0], "mat_mul");

        int M = output_tensor->dims[1];
        int N = output_tensor->dims[2] * output_tensor->dims[3];
        int K = weight_tensor->dims[1] * weight_tensor->dims[2] * weight_tensor->dims[3];

        arg_idx = 0;
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&clBuffer) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[weight_tensor->idx]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->idx]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &M) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &N) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &K) );

        struct OCLqueue Mat_Mul;
        Mat_Mul.name = "Mat_Mul"; //std::to_string(ir_node->idx); //"Mat_Mul";
        Mat_Mul.dims = 1;
        Mat_Mul.queue_kernel = this->kernel;
        Mat_Mul.queue_global_work_size = (size_t*)malloc(1 * sizeof(size_t));
        Mat_Mul.queue_global_work_size[0] = ((M + 3)/4*4) * ((N + 3)/4*4);
        Mat_Mul.queue_local_work_size = (size_t*)malloc(1 * sizeof(size_t));
        Mat_Mul.queue_local_work_size[0] = (M + 3)/4;
        this->queue_list.push_back(Mat_Mul);

        /* bias_add and relu compute */
        if (2 < ir_node->input_num)
        {
            int elem_23 = output_tensor->dims[2] * output_tensor->dims[3];
            int elem_123 = output_tensor->dims[1] * elem_23;

            if (conv_param->activation == 0)
            {
                char cl_kernel_path2[500] = "";
                strcat(cl_kernel_path2, cl_env);
                strcat(cl_kernel_path2, "/src/dev/opencl/cl/bias_add_relu.cl");
                //fprintf(stderr,"Log cl kernel path: %s\n",cl_kernel_path2);
                this->build_kernel(&cl_kernel_path2[0], "bias_add_relu");

                arg_idx = 0;
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->idx]) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[bias_tensor->idx]) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_123) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_23) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

                struct OCLqueue Bias_Add_Relu;
                Bias_Add_Relu.name = "Bias_Add_Relu";
                Bias_Add_Relu.dims = 1;
                Bias_Add_Relu.queue_kernel = this->kernel;
                Bias_Add_Relu.queue_global_work_size = (size_t*)malloc(1 * sizeof(size_t));
                Bias_Add_Relu.queue_global_work_size[0] = ((M + 3)/4*4) * ((N + 3)/4*4);
                Bias_Add_Relu.queue_local_work_size = (size_t*)malloc(1 * sizeof(size_t));
                Bias_Add_Relu.queue_local_work_size[0] = (M + 3)/4;
                this->queue_list.push_back(Bias_Add_Relu);

            }
            else
            {
                char cl_kernel_path2[500] = "";
                strcat(cl_kernel_path2, cl_env);
                strcat(cl_kernel_path2, "/src/dev/opencl/cl/bias_add.cl");
                //fprintf(stderr,"Log cl kernel path: %s\n",cl_kernel_path2);
                this->build_kernel(&cl_kernel_path2[0], "bias_add");

                arg_idx = 0;
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->idx]) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[bias_tensor->idx]) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_123) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_23) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

                struct OCLqueue Bias_Add;
                Bias_Add.name = "Bias_Add";
                Bias_Add.dims = 1;
                Bias_Add.queue_kernel = this->kernel;
                Bias_Add.queue_global_work_size = (size_t*)malloc(1 * sizeof(size_t));
                Bias_Add.queue_global_work_size[0] = ((M + 3)/4*4) * ((N + 3)/4*4);
                Bias_Add.queue_local_work_size = (size_t*)malloc(1 * sizeof(size_t));
                Bias_Add.queue_local_work_size[0] = (M + 3)/4;
                this->queue_list.push_back(Bias_Add);

            }
        }
    }
    else // DW Conv
    {
        char* cl_env = getenv("ROOT_PATH");
        char cl_kernel_path[500] = "";
        strcat(cl_kernel_path, cl_env);
        strcat(cl_kernel_path, "/src/dev/opencl/cl/depthwise_conv.cl");
        //fprintf(stderr,"Log cl kernel path: %s\n",cl_kernel_path);
        this->build_kernel(&cl_kernel_path[0], "depthwise_conv");

        int arg_idx = 0;
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(int), &output_tensor->elem_num) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->idx]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[weight_tensor->idx]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &input_tensor->dims[1]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &input_tensor->dims[2]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &input_tensor->dims[3]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->dims[2]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->dims[3]) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->kernel_h) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->kernel_w) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->stride_h) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->stride_w) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->pad_h0) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &conv_param->pad_w0) );
        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->idx]) );

        struct OCLqueue DWconv;
        DWconv.name = "DWconv";
        DWconv.dims = 1;
        DWconv.queue_kernel = this->kernel;
        DWconv.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
        DWconv.queue_global_work_size[0] = output_tensor->elem_num;
        DWconv.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
        DWconv.queue_local_work_size[0] =  1;
        this->queue_list.push_back(DWconv);

        int M = output_tensor->dims[1];
        int N = output_tensor->dims[2] * output_tensor->dims[3];
        int K = weight_tensor->dims[1] * weight_tensor->dims[2] * weight_tensor->dims[3];

        /* bias_add and relu compute */
        if (2 < ir_node->input_num)
        {
            int elem_23 = output_tensor->dims[2] * output_tensor->dims[3];
            int elem_123 = output_tensor->dims[1] * elem_23;

            if (conv_param->activation == 0)
            {
                char cl_kernel_path2[500] = "";
                strcat(cl_kernel_path2, cl_env);
                strcat(cl_kernel_path2, "/src/dev/opencl/cl/bias_add_relu.cl");
                //fprintf(stderr,"Log cl kernel path: %s\n",cl_kernel_path2);
                this->build_kernel(&cl_kernel_path2[0], "bias_add_relu");

                arg_idx = 0;
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->idx]) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[bias_tensor->idx]) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_123) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_23) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

                struct OCLqueue Bias_Add_Relu;
                Bias_Add_Relu.name = "Bias_Add_Relu";
                Bias_Add_Relu.dims = 1;
                Bias_Add_Relu.queue_kernel = this->kernel;
                Bias_Add_Relu.queue_global_work_size = (size_t*)malloc(1 * sizeof(size_t));
                Bias_Add_Relu.queue_global_work_size[0] = ((M + 3)/4*4) * ((N + 3)/4*4);
                Bias_Add_Relu.queue_local_work_size = (size_t*)malloc(1 * sizeof(size_t));
                Bias_Add_Relu.queue_local_work_size[0] = (M + 3)/4;
                this->queue_list.push_back(Bias_Add_Relu);

            }
            else
            {
                char cl_kernel_path2[500] = "";
                strcat(cl_kernel_path2, cl_env);
                strcat(cl_kernel_path2, "/src/dev/opencl/cl/bias_add.cl");
                //fprintf(stderr,"Log cl kernel path: %s\n",cl_kernel_path2);
                this->build_kernel(&cl_kernel_path2[0], "bias_add");

                arg_idx = 0;
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel,   arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->idx]) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[bias_tensor->idx]) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_123) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &elem_23) );
                CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num) );

                struct OCLqueue Bias_Add;
                Bias_Add.name = "Bias_Add";
                Bias_Add.dims = 1;
                Bias_Add.queue_kernel = this->kernel;
                Bias_Add.queue_global_work_size = (size_t*)malloc(1 * sizeof(size_t));
                Bias_Add.queue_global_work_size[0] = ((M + 3)/4*4) * ((N + 3)/4*4);
                Bias_Add.queue_local_work_size = (size_t*)malloc(1 * sizeof(size_t));
                Bias_Add.queue_local_work_size[0] = (M + 3)/4;
                this->queue_list.push_back(Bias_Add);

            }
        }
    }



    return true;
}












