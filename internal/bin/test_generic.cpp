#include <iostream>
#include <string>
#include <typeinfo>

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

int create_generic_node(graph_t graph, const char* node_name, const char* input_name)
{
    node_t generic_node = create_graph_node(graph, node_name, "Generic");

    if(generic_node == nullptr)
    {
        std::cerr << "create node failed. errno " << get_tengine_errno() << "\n";
        return -1;
    }

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(generic_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);

    set_node_output_tensor(generic_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    add_node_attr(generic_node, "alpha", typeid(float).name(), sizeof(float));
    add_node_attr(generic_node, "beta", typeid(float).name(), sizeof(float));

    int input_num = 1;
    int output_num = 2;

    const char* op_name = "MUL";

    if(set_node_attr_pointer(generic_node, "op_name", &op_name) < 0)
    {
        std::cerr << "set node attr failed\n";
    }
    set_node_attr_int(generic_node, "max_input_num", &input_num);
    set_node_attr_int(generic_node, "max_output_num", &output_num);

    float val = 2;
    set_node_attr_float(generic_node, "alpha", &val);
    val = 10000;
    set_node_attr_float(generic_node, "beta", &val);

    release_graph_node(generic_node);

    return 0;
}

graph_t create_test_graph(int c, int h, int w)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* conv_name = "conv";
    const char* generic_name = "generic";

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

    /* add the generic node */

    if(create_generic_node(graph, generic_name, conv_name))
    {
        std::cerr << "create generic node failed\n";
        return nullptr;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {generic_name};

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
static struct custom_kernel_ops generic_custom_kernel;

struct my_op_param
{
    float alpha;
    float beta;
};

static my_op_param op_param;

int my_infer_shape(struct custom_kernel_ops* ops, const int* inputs[], int input_num, int* outputs[], int output_num,
                   int layout)
{
    std::cout << __FUNCTION__ << " called\n";
    std::cout << "input_num " << input_num << "\n";
    std::cout << "output_num " << output_num << "\n";

    int* output_shape = outputs[0];
    const int* input_shape = inputs[0];

    for(int i = 0; i < MAX_SHAPE_DIM_NUM; i++)
        output_shape[i] = input_shape[i];

    return 0;
}

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
    std::cout << "input_num " << input_num << "\n";
    std::cout << "output_num " << output_num << "\n";

    float* input_data = ( float* )inputs[0]->data;
    float* output_data = ( float* )outputs[0]->data;
    my_op_param* param = ( my_op_param* )ops->kernel_param;
    float alpha = param->alpha;
    float beta = param->beta;

    for(int i = 0; i < inputs[0]->element_num; i++)
    {
        output_data[i] = input_data[i] * alpha + beta;
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
    ops->op = "MUL";
    ops->force = 1;

    ops->infer_shape = my_infer_shape;
    ops->inplace_info = nullptr;
    ops->bind = nullptr;
    ops->reshape = nullptr;
    ops->prerun = my_prerun;
    ops->run = my_run;
    ops->postrun = my_postrun;
    ops->release = my_release;
    ops->kernel_param = &op_param;
    ops->kernel_param_size = sizeof(op_param);
}

void fill_custom_kernel_param(node_t node, struct custom_kernel_ops* ops)
{
    my_op_param* param = ( my_op_param* )ops->kernel_param;

    if(get_node_attr_float(node, "alpha", &param->alpha) < 0)
    {
        std::cerr << "get attr alpha failed\n";
    }

    if(get_node_attr_float(node, "beta", &param->beta) < 0)
    {
        std::cerr << "get attr beta failed\n";
    }
}

int main(int argc, char* argv[])
{
    int c, h, w;

    c = 3;
    h = 16;
    w = 16;

    init_tengine();

    graph_t graph = create_test_graph(c, h, w);

    if(graph == nullptr)
        return 1;

    /* using custom kernel to do cacluation */
    init_custom_kernel(&generic_custom_kernel);

    node_t my_node = get_graph_node(graph, "generic");

    fill_custom_kernel_param(my_node, &generic_custom_kernel);

    if(set_custom_kernel(my_node, "ANY_DEVICE", &generic_custom_kernel) < 0)
    {
        std::cerr << "set_custom_kernel failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    release_graph_node(my_node);

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

    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    dump_graph(graph);

    const char* dev = get_node_device(conv_node);

    std::cout << "node running on dev: " << dev << "\n";

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

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
