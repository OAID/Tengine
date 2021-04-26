

#include "cuda_executor.hpp"

extern "C"
{
#include "convolution_param.h"

#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

__global__ void bias_add(float *y, float *x, int elem_num_perimg, int elem_perchannel, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        y[idx] += x[idx % elem_num_perimg / elem_perchannel];
    }
}

__global__ void bias_add_relu(float *y, float *x, int elem_num_perimg, int elem_perchannel, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        y[idx] += x[idx % elem_num_perimg / elem_perchannel];
        y[idx] = y[idx] > 0 ? y[idx] : 0;
    }
}

void conv_gpu_kernel(cudnnHandle_t& handle, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx  gpu_addr_map,
                     cudnnConvolutionFwdAlgo_t& algo1, int setalgo)
{
    struct tensor* conv_input_data = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* conv_weight = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* conv_output_data = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    // input
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               conv_input_data->dims[0], conv_input_data->dims[1], conv_input_data->dims[2], conv_input_data->dims[3]);

    // output
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               conv_output_data->dims[0], conv_output_data->dims[1], conv_output_data->dims[2], conv_output_data->dims[3]);

    // kernel
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               conv_weight->dims[0], conv_weight->dims[1], conv_weight->dims[2], conv_weight->dims[3]);

    // convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor,
                                    conv_param->pad_h0, conv_param->pad_w0, // zero-padding
                                    conv_param->stride_h, conv_param->stride_w, // stride
                                    conv_param->dilation_h, conv_param->dilation_w,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(conv_descriptor, conv_param->group);

//    // algorithm
//    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
//
//    // workspace size && allocate memory
//    size_t workspace_size;
//    cudnnGetConvolutionForwardWorkspaceSize(
//            handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, algo, &workspace_size);
//
//    size_t workspace_size;
//    cudnnGetConvolutionForwardWorkspaceSize(handle,
//                                            input_descriptor,
//                                            kernel_descriptor,
//                                            conv_descriptor,
//                                            output_descriptor,
//                                            algo,
//                                            &workspace_size);
//
//    void * workspace = nullptr;
//    cudaMalloc(&workspace, workspace_size);

    if (0 == setalgo)
    {
        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerf_t algo;
        auto ret0 = cudnnFindConvolutionForwardAlgorithm(handle,
                                                         input_descriptor,
                                                         kernel_descriptor,
                                                         conv_descriptor,
                                                         output_descriptor,
                                                         1,
                                                         &returnedAlgoCount,
                                                         &algo
        );
        algo1 = algo.algo;
    }
    size_t workspace_size;
    auto ret1 = cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, algo1, &workspace_size);
    void * workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);


//    size_t workspace_size = 0;
//    void * workspace = nullptr;
//    cudaMalloc(&workspace, workspace_size);
//    if (0 == setalgo)
//    {
//        int returnedAlgoCount;
//        cudnnConvolutionFwdAlgoPerf_t algo;
//        auto ret0 = cudnnFindConvolutionForwardAlgorithmEx(handle,
//                                                           input_descriptor, gpu_addr_map[conv_input_data->index],
//                                                           kernel_descriptor, gpu_addr_map[conv_weight->index],
//                                                           conv_descriptor,
//                                                           output_descriptor, gpu_addr_map[conv_output_data->index],
//                                                           1,
//                                                           &returnedAlgoCount,
//                                                           &algo,
//                                                           workspace,
//                                                           workspace_size
//                                                           );
//        algo1 = algo.algo;
//    }


    /* convolution forward run */
    auto alpha = 1.0f, beta = 0.0f;
    auto ret2 = cudnnConvolutionForward(handle,
                                        &alpha, input_descriptor, gpu_addr_map[conv_input_data->index],
                                        kernel_descriptor, gpu_addr_map[conv_weight->index],
                                        conv_descriptor, algo1,
                                        workspace, workspace_size,
                                        &beta, output_descriptor, gpu_addr_map[conv_output_data->index]);

    /* init grid and block */
    int bs = 1024;
    int s = ceil((conv_output_data->elem_num + bs - 1.) / bs);
    dim3 grid = dim3(s);

    /* bias_add and relu compute */
    if (2 < ir_node->input_num)
    {
        struct tensor* conv_bias = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        int elem_23 = conv_output_data->dims[2] * conv_output_data->dims[3];
        int elem_123 = conv_output_data->dims[1] * elem_23;

        if (conv_param->activation == 0)
            bias_add_relu<<<grid, bs>>>((float*)gpu_addr_map[conv_output_data->index], (float*)gpu_addr_map[conv_bias->index], elem_123, elem_23, conv_output_data->elem_num);
        else
            bias_add<<<grid, bs>>>((float*)gpu_addr_map[conv_output_data->index], (float*)gpu_addr_map[conv_bias->index], elem_123, elem_23, conv_output_data->elem_num);
    }

    cudaFree(&workspace);
}



void CUDAEngine::AddConvolutionNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_CONV.\n", ir_node->index);
    cudnnCreate(&this->cudnn_handle);
    conv_gpu_kernel(this->cudnn_handle, ir_graph, ir_node, this->gpu_addr_map, this->algo1, 0);
    this->ops.push_back(std::bind(&conv_gpu_kernel, this->cudnn_handle, ir_graph, ir_node, this->gpu_addr_map, this->algo1, 1));
}
