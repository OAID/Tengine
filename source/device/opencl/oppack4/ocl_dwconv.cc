
#include <memory>
#include "ocl_dwconv.hpp"
#include "ocl_convertor.hpp"
#include "ocl_executor.hpp"

void ocl_dwconv::upload_bias_gpu(struct tensor* ir_tensor)
{
    int bias_size = ir_tensor->elem_num;
    int buffer_size = ROUND_UP(bias_size, 4);
    cl::Buffer bias_buffer(engine->get_context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bias_size * sizeof(float));
    cl_int error;
    auto* bias_ptr_gpu = (float*)engine->get_command_queue().enqueueMapBuffer(bias_buffer, true, CL_MAP_WRITE, 0, bias_size * sizeof(float), nullptr, nullptr, &error);
    if (bias_ptr_gpu != nullptr && error == CL_SUCCESS)
    {
        ::memset(bias_ptr_gpu, 0, bias_size * sizeof(float));
        ::memcpy(bias_ptr_gpu, ir_tensor->data, bias_size * sizeof(float));
    }
    engine->get_command_queue().enqueueUnmapMemObject(bias_buffer, bias_ptr_gpu);
    gpu_bias = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), UP_DIV(bias_size, 4), 1);
    engine->get_converter().buffer_to_image(&bias_buffer, gpu_bias.get(), UP_DIV(bias_size, 4), 1);
}

ocl_dwconv::ocl_dwconv(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* bias_tensor;
    if (2 < ir_node->input_num)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        upload_bias_gpu(bias_tensor);
    }

    auto* conv_2d_param = (struct conv_param*)ir_node->op.param_mem;
    this->conv2d_param = conv_2d_param;
    strides = {conv_2d_param->stride_h, conv_2d_param->stride_w};
    dilations = {conv_2d_param->dilation_h, conv_2d_param->dilation_w};
    paddings = {conv_2d_param->pad_h0, conv_2d_param->pad_w0};
    int kernel_width = conv_2d_param->kernel_w;
    int kernel_height = conv_2d_param->kernel_h;
    int out_channel = conv_2d_param->output_channel;

    int filter_size = kernel_height * kernel_width * out_channel;
    int filter_buffer_size = filter_size * sizeof(float);
    cl::Buffer filter_buffer(engine->get_context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, filter_buffer_size);
    cl_int error;
    auto filter_ptr_gpu = engine->get_command_queue().enqueueMapBuffer(filter_buffer, true, CL_MAP_WRITE, 0, filter_buffer_size, nullptr, nullptr, &error);
    if (filter_ptr_gpu != nullptr && error == CL_SUCCESS)
    {
        ::memset(filter_ptr_gpu, 0, filter_buffer_size);
        ::memcpy(filter_ptr_gpu, weight_tensor->data, filter_buffer_size);
    }
    else
    {
        TLOG_ERR("error in filter_ptr_gpu\n");
    }
    engine->get_command_queue().enqueueUnmapMemObject(filter_buffer, filter_ptr_gpu);
    gpu_weight = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), kernel_width * kernel_height, UP_DIV(out_channel, 4));
    engine->get_converter().dw_filter_buffer_to_image(conv_2d_param, &filter_buffer, gpu_weight.get());

    std::set<std::string> buildOption;
    if (2 >= ir_node->input_num)
    {
        buildOption.emplace("-DNO_BIAS");
    }
    if (conv2d_param->activation == 0)
    {
        buildOption.emplace("-DRELU");
    }
    else if (conv_2d_param->activation == 6)
    {
        buildOption.emplace("-DRELU6");
    }

    std::string kernelName = "depthwise_conv2d";
    if (conv_2d_param->stride_w == 1 && conv_2d_param->stride_h == 1)
    {
        kernelName = "depthwise_conv2d_s1";
    }
    conv2d_kernel = engine->build_kernel("depthwise_conv2d", kernelName, buildOption);
    max_work_group_size = (int)engine->get_max_work_group_size(conv2d_kernel);
}

void ocl_dwconv::pre_run()
{
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_output = ir_node->output_tensors[0];

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_output);

    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_tensor_idx_output);

    // TLOG_ERR("handle_input: %lld  handle_output: %lld \n", handle_input, handle_output);

    int height = output_tensor->dims[2];
    int width = output_tensor->dims[3];
    int input_height = input_tensor->dims[2];
    int input_width = input_tensor->dims[3];
    int input_channel = input_tensor->dims[1];
    int input_channel_block = UP_DIV(input_channel, 4);
    int output_channel = output_tensor->dims[1];
    int output_height = output_tensor->dims[2];
    int kernel_width = this->conv2d_param->kernel_w;
    int kernel_height = this->conv2d_param->kernel_h;
    int global_0 = UP_DIV(width, 4) * UP_DIV(output_channel, 4);
    int global_1 = output_height;

    int input_image_shape[2] = {input_height, input_width};
    int output_image_shape[2] = {height, width};
    int kernel_shape[2] = {kernel_height, kernel_width};
    int stride_shape[2] = {strides[0], strides[1]};
    int padding_shape[2] = {paddings[0], paddings[1]};
    int dilation_shape[2] = {dilations[0], dilations[1]};

    uint32_t idx = 0;
    auto kernel = &conv2d_kernel;
    kernel->setArg(idx++, global_0);
    kernel->setArg(idx++, global_1);

    kernel->setArg(idx++, *(cl::Image*)handle_input);
    kernel->setArg(idx++, *gpu_weight);
    if (2 < ir_node->input_num)
    {
        kernel->setArg(idx++, *gpu_bias);
    }
    kernel->setArg(idx++, *(cl::Image*)handle_output);
    kernel->setArg(idx++, sizeof(input_image_shape), input_image_shape);
    kernel->setArg(idx++, input_channel_block);
    kernel->setArg(idx++, sizeof(output_image_shape), output_image_shape);
    kernel->setArg(idx++, sizeof(kernel_shape), kernel_shape);
    kernel->setArg(idx++, sizeof(padding_shape), padding_shape);
    if (conv2d_param->stride_w != 1 && conv2d_param->stride_h != 1)
    {
        kernel->setArg(idx++, sizeof(dilation_shape), dilation_shape);
        kernel->setArg(idx, sizeof(stride_shape), stride_shape);
    }

    global_work_size = {(uint32_t)global_0, (uint32_t)global_1};
    local_work_size = find_local_group_2d(global_work_size, max_work_group_size, engine, conv2d_kernel, ir_node->name);
}

void ocl_dwconv::run(struct subgraph* subgraph)
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
    run_node_2d(global_work_size, local_work_size, conv2d_kernel, &event);
    int cost = (int)engine->get_cost_time(&event);
    TLOG_ERR("cost: %d dwconv:%s \n", cost, ir_node->name);
#else
    run_node_2d(global_work_size, local_work_size, conv2d_kernel);
#endif
#if 0

#endif

    //debug_data();
}
