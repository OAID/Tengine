#include <iostream>
#include <string>

#include "tengine_c_api.h"

context_t shared_context = nullptr;

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

    int weight_size = out_c * in_c * k_size * k_size * sizeof(float);
    void* weight_data = malloc(weight_size);
    set_tensor_buffer(w_tensor, weight_data, weight_size);

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
    graph_t graph = create_graph(shared_context, nullptr, nullptr);

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

void create_pooling_node(graph_t graph, const char* node_name, const char* input_name)
{
    node_t pool_node = create_graph_node(graph, node_name, "Pooling");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return;
    }

    set_node_input_tensor(pool_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(pool_node, 0, output_tensor, TENSOR_TYPE_VAR);
    release_graph_tensor(output_tensor);

    release_graph_node(pool_node);
}

graph_t create_pooling_graph(const char* input_name)
{
    const char* pool_name = "pool1";

    graph_t graph = create_graph(shared_context, nullptr, nullptr);

    create_input_node(graph, input_name, 64, 112, 112);

    create_pooling_node(graph, pool_name, input_name);

    const char* inputs[] = {input_name};

    const char* outputs[] = {pool_name};

    set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*));

    set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*));

    return graph;
}

graph_t create_pooling_graph2(const char* input_name)
{
    const char* pool_name = "pool2";

    graph_t graph = create_graph(shared_context, nullptr, nullptr);

    /* create input tensor */

    tensor_t input_tensor = create_graph_tensor(graph, input_name, TENGINE_DT_FP32);

    if(input_tensor == nullptr)
    {
        std::cout << __FUNCTION__ << " ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    release_graph_tensor(input_tensor);

    create_pooling_node(graph, pool_name, input_name);

    const char* inputs[] = {pool_name};

    const char* outputs[] = {pool_name};

    set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*));

    set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*));

    return graph;
}

int main(int argc, char* argv[])
{
    int c, h, w;

    c = 3;
    h = 16;
    w = 16;

    init_tengine();

    shared_context = create_context("shared", 0);

    graph_t graph0 = create_conv_graph(c, h, w);

    if(graph0 == nullptr)
        return 1;

    graph_t graph1 = create_pooling_graph("conv");

    graph_t m_graph = merge_graph(2, graph0, graph1);

    if(prerun_graph(m_graph) < 0)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    dump_graph(m_graph);
    // second test
    std::cout << "Merge test 2\n";
    graph_t graph2 = create_conv_graph(c, h, w);
    graph_t graph3 = create_pooling_graph2("conv");

    graph_t m_graph2 = merge_graph(2, graph2, graph3);

    if(m_graph2 == nullptr)
    {
        std::cout << "create merge 2 failed\n";
        return 1;
    }

    if(prerun_graph(m_graph2) < 0)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    std::cout << "\n\n=====\n\n";
    dump_graph(m_graph2);

    postrun_graph(m_graph);
    postrun_graph(m_graph2);

    destroy_graph(m_graph);
    destroy_graph(m_graph2);
    destroy_graph(graph0);
    destroy_graph(graph1);
    destroy_graph(graph2);
    destroy_graph(graph3);
    destroy_context(shared_context);

    release_tengine();
    return 0;
}
