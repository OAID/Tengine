#include <iostream>
#include <string>

/* the sample code to create a convolution and do calculation */

#include "tengine_c_api.h"

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int k_size, int stride, int pad,
                     int in_c, int out_c, int group)
{
    node_t conv_node = create_graph_node(graph, node_name, "Convolution");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(conv_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);

    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* weight */

    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);

    set_node_input_tensor(conv_node, 1, w_tensor);
    int w_dims[] = {out_c, in_c, k_size, k_size};

    set_tensor_shape(w_tensor, w_dims, 4);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    /* bias */
    std::string bias_name(node_name);
    bias_name += "/bias";

    node_t b_node = create_graph_node(graph, bias_name.c_str(), "Const");
    tensor_t b_tensor = create_graph_tensor(graph, bias_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);

    int b_dims[] = {64};

    set_tensor_shape(b_tensor, b_dims, 1);

    set_node_input_tensor(conv_node, 2, b_tensor);
    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    /* attr */
    set_node_attr_int(conv_node, "kernel_h", &k_size);
    set_node_attr_int(conv_node, "kernel_w", &k_size);
    set_node_attr_int(conv_node, "stride_h", &stride);
    set_node_attr_int(conv_node, "stride_w", &stride);
    set_node_attr_int(conv_node, "pad_h", &pad);
    set_node_attr_int(conv_node, "pad_w", &pad);
    set_node_attr_int(conv_node, "output_channel", &out_c);
    set_node_attr_int(conv_node, "group", &group);

    release_graph_node(conv_node);

    return 0;
}

graph_t create_conv_graph(int c, int h, int w)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* conv_name = "conv";

    if(create_input_node(graph, input_name, c, h, w) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_conv_node(graph, conv_name, input_name, 3, 1, 1, c, 64, 1) < 0)
    {
        std::cerr << "create conv node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {conv_name};

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set outputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    return graph;
}

/* custom kernel impl */

static struct custom_kernel_ops conv_custom_kernel;

struct my_conv_param
{
    int kernel_h;
    int kernel_w;
    int pad_h0;
    int pad_h1;
    int pad_w0;
    int pad_w1;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int in_channel;
    int out_channel;
    int group;
};

static my_conv_param conv_param;

int my_prerun(struct custom_kernel_ops* ops, struct custom_kernel_tensor* inputs[], int input_num,
              struct custom_kernel_tensor* outputs[], int output_num, int dynamic_shape)
{
    std::cout << __FUNCTION__ << " called\n";
    std::cout << "input_num " << input_num << "\n";
    std::cout << "output_num " << output_num << "\n";

    return 0;
}

int my_run(struct custom_kernel_ops* ops, struct custom_kernel_tensor* inputs[], int input_num,
           struct custom_kernel_tensor* outputs[], int output_num)
{
    std::cout << __FUNCTION__ << " called\n";

    my_conv_param* param = ( my_conv_param* )ops->kernel_param;

    struct custom_kernel_tensor* output = outputs[0];
    struct custom_kernel_tensor* input = inputs[0];
    struct custom_kernel_tensor* weight = inputs[1];
    struct custom_kernel_tensor* bias = nullptr;

    if(input_num > 2)
        bias = inputs[2];

    int n = output->dim[0];
    int out_c = output->dim[1];
    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;

    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;

    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];

    float* src_ptr = ( float* )input->data;
    float* dst_ptr = ( float* )output->data;

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < out_c; j++)
        {
            float bias_data = 0;

            if(bias)
            {
                float* bias_ptr = ( float* )bias->data;
                bias_data = bias_ptr[j];
            }

            for(int h = 0; h < out_h; h++)
                for(int w = 0; w < out_w; w++)
                {
                    float* weight_ptr = ((( float* )weight->data) + j * in_c * kernel_h * kernel_w);
                    float sum = 0;

                    for(int c = 0; c < in_c; c++)
                    {
                        int start_h = h * stride_h - pad_h0;
                        int start_w = w * stride_w - pad_w0;
                        float* src_hw = src_ptr + c * in_h * in_w;
                        float* weight_base = weight_ptr + c * kernel_h * kernel_w;

                        /* one slide window */
                        for(int k_h = 0; k_h < kernel_h; k_h++)
                        {
                            int src_h = start_h + k_h;

                            if(src_h < 0 || src_h >= in_h)
                                continue;

                            for(int k_w = 0; k_w < kernel_w; k_w++)
                            {
                                int src_w = start_w + k_w;

                                if(src_w < 0 || src_w >= in_w)
                                    continue;

                                float wt = weight_base[k_h * kernel_w + k_w];
                                float src = src_hw[src_h * in_w + src_w];

                                sum += wt * src;
                            }
                        }
                    }

                    sum += bias_data;

                    dst_ptr[h * out_w + w] = sum;
                }

            dst_ptr += out_h * out_w;
        }

        src_ptr += in_c * in_h * in_w;
    }

    return 0;
}

int my_postrun(struct custom_kernel_ops* ops, struct custom_kernel_tensor* inputs[], int input_num,
               struct custom_kernel_tensor* outputs[], int output_num)
{
    std::cout << __FUNCTION__ << " called\n";
    std::cout << "input_num " << input_num << "\n";
    std::cout << "output_num " << output_num << "\n";

    return 0;
}

void my_release(struct custom_kernel_ops* ops)
{
    std::cout << __FUNCTION__ << " called\n";
}

void init_custom_kernel(struct custom_kernel_ops* ops)
{
    ops->kernel_name = "custom_conv";
    ops->op = "Convolution";
    ops->force = 1;

    ops->infer_shape = nullptr;
    ops->inplace_info = nullptr;
    ops->bind = nullptr;
    ops->reshape = nullptr;
    ops->prerun = my_prerun;
    ops->run = my_run;
    ops->postrun = my_postrun;
    ops->release = my_release;
    ops->kernel_param = &conv_param;
    ops->kernel_param_size = sizeof(conv_param);
}

void fill_custom_kernel_param(node_t node, struct custom_kernel_ops* ops)
{
    my_conv_param* param = ( my_conv_param* )ops->kernel_param;

    get_node_attr_int(node, "kernel_h", &param->kernel_h);
    get_node_attr_int(node, "kernel_w", &param->kernel_w);
    get_node_attr_int(node, "stride_h", &param->stride_h);
    get_node_attr_int(node, "stride_w", &param->stride_w);
    get_node_attr_int(node, "dilation_h", &param->dilation_h);
    get_node_attr_int(node, "dilation_w", &param->dilation_w);
    get_node_attr_int(node, "pad_h", &param->pad_h0);
    get_node_attr_int(node, "pad_w", &param->pad_w0);
    get_node_attr_int(node, "output_channel", &param->out_channel);
    get_node_attr_int(node, "group", &param->group);

    param->pad_h1 = param->pad_h0;
    param->pad_w1 = param->pad_w0;
}

int main(int argc, char* argv[])
{
    int c, h, w;

    c = 3;
    h = 16;
    w = 16;

    init_tengine();

    graph_t graph = create_conv_graph(c, h, w);

    if(graph == nullptr)
        return 1;

    /* using custom kernel to do cacluation */
    init_custom_kernel(&conv_custom_kernel);

    node_t my_node = get_graph_node(graph, "conv");

    fill_custom_kernel_param(my_node, &conv_custom_kernel);

    if(set_custom_kernel(my_node, "ANY_DEVICE", &conv_custom_kernel) < 0)
    {
        std::cerr << "set_custom_kernel failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    release_graph_node(my_node);

    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
        i_buf[i] = 1;

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);

    /* set weight */
    node_t conv_node = get_graph_node(graph, "conv");

    tensor_t weight_tensor = get_node_input_tensor(conv_node, 1);

    buf_size = get_tensor_buffer_size(weight_tensor);
    float* w_buf = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
        w_buf[i] = 1;

    set_tensor_buffer(weight_tensor, w_buf, buf_size);

    release_graph_tensor(weight_tensor);

    /* set bias */

    int input_num = get_node_input_number(conv_node);
    float* b_buf = nullptr;

    if(input_num > 2)
    {
        tensor_t bias_tensor = get_node_input_tensor(conv_node, 2);

        buf_size = get_tensor_buffer_size(bias_tensor);
        b_buf = ( float* )malloc(buf_size);

        for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
            b_buf[i] = 1;

        set_tensor_buffer(bias_tensor, b_buf, buf_size);
        release_graph_tensor(bias_tensor);
    }

    // dump_graph(graph);

    const char* dev = get_node_device(conv_node);

    std::cout << "node running on dev: " << dev << "\n";

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }

    tensor_t output_tensor = get_node_output_tensor(conv_node, 0);

    float* buf = ( float* )get_tensor_buffer(output_tensor);

    for(int i = 0; i < w; i++)
        std::cout << i << " " << buf[i] << "\n";

    release_graph_tensor(output_tensor);
    release_graph_node(conv_node);

    postrun_graph(graph);

    destroy_graph(graph);

    free(i_buf);
    free(w_buf);

    if(b_buf)
        free(b_buf);

    release_tengine();

    std::cout << "TEST DONE\n";
    return 0;
}
