#include <iostream>
#include <string>

/* the sample code to create a convolution and do calculation */

#include "tengine_c_api.h"

int int8_mismatch(int8_t* a, int8_t* b, int size)
{
    int i = 0;
    for(i = 0; i < size; i++)
    {
        int8_t off = a[i] - b[i];
        if(off != 0)
            break;
    }
    if(i != size)
    {
        printf("mismatch:\n\t[%d]\t---a:    %d ,%d   :b---        off: %d\n", i, a[i], b[i], a[i] - b[i]);
        return -1;
    }
    return 0;
}

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_INT8);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int k_size, int stride, int pad_h,
                     int pad_w, int k_d, int in_c, int out_c, int group)
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
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_INT8);
    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* weight */

    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_INT8);
    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    set_node_input_tensor(conv_node, 1, w_tensor);
    int w_dims[] = {out_c, in_c / group, k_size, k_size};

    set_tensor_shape(w_tensor, w_dims, 4);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    /* bias */
    std::string bias_name(node_name);
    bias_name += "/bias";

    node_t b_node = create_graph_node(graph, bias_name.c_str(), "Const");
    tensor_t b_tensor = create_graph_tensor(graph, bias_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
    int b_dims[] = {out_c};

    set_tensor_shape(b_tensor, b_dims, 1);

    set_node_input_tensor(conv_node, 2, b_tensor);
    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    /* attr */
    set_node_attr_int(conv_node, "kernel_h", &k_size);
    set_node_attr_int(conv_node, "kernel_w", &k_size);
    set_node_attr_int(conv_node, "stride_h", &stride);
    set_node_attr_int(conv_node, "stride_w", &stride);
    set_node_attr_int(conv_node, "pad_h", &pad_h);
    set_node_attr_int(conv_node, "pad_w", &pad_w);
    set_node_attr_int(conv_node, "dilation_h", &k_d);
    set_node_attr_int(conv_node, "dilation_w", &k_d);
    set_node_attr_int(conv_node, "output_channel", &out_c);
    set_node_attr_int(conv_node, "group", &group);

    release_graph_node(conv_node);

    return 0;
}

graph_t create_conv_graph(int c, int h, int w, int k, int s, int p_h, int p_w, int d, int o, int g)
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

    if(create_conv_node(graph, conv_name, input_name, k, s, p_h, p_w, d, c, o, g) < 0)
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

int test_conv(int c, int h, int w, int k, int s, int p_h, int p_w, int d, int o, int g)
{
    //    std::cout<<"in_c:"<< c <<",input_h:"<<h <<",input_w:"<<w<<",stride:"<<s<<",p_h:"<<p_h<<",p_w:" <<p_w<<"\n";

    graph_t graph = create_conv_graph(c, h, w, k, s, p_h, p_w, d, o, g);
    graph_t graph1 = create_conv_graph(c, h, w, k, s, p_h, p_w, d, o, g);
    if(graph == nullptr || graph1 == nullptr)
        return 1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t input_tensor1 = get_graph_input_tensor(graph1, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    int8_t* i_buf = ( int8_t* )malloc(buf_size);
    int8_t* i_buf1 = ( int8_t* )malloc(buf_size);

    for(int i = 0; i < buf_size; i++)
    {
        i_buf[i] = 1;
        i_buf1[i] = 1;
    }

    float scale = 0.1;
    int zero = 0;
    set_tensor_quant_param(input_tensor, &scale, &zero, 1);
    set_tensor_quant_param(input_tensor1, &scale, &zero, 1);

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    set_tensor_buffer(input_tensor1, i_buf1, buf_size);
    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor1);

    /* set weight */
    node_t conv_node = get_graph_node(graph, "conv");
    node_t conv_node1 = get_graph_node(graph1, "conv");

    tensor_t weight_tensor = get_node_input_tensor(conv_node, 1);
    tensor_t weight_tensor1 = get_node_input_tensor(conv_node1, 1);

    set_tensor_quant_param(weight_tensor, &scale, &zero, 1);
    set_tensor_quant_param(weight_tensor1, &scale, &zero, 1);

    buf_size = get_tensor_buffer_size(weight_tensor);
    int8_t* w_buf = ( int8_t* )malloc(buf_size);
    int8_t* w_buf1 = ( int8_t* )malloc(buf_size);

    for(int i = 0; i < buf_size; i++)
    {
        w_buf[i] = 1;
        w_buf1[i] = 1;
    }

    set_tensor_buffer(weight_tensor, w_buf, buf_size);
    set_tensor_buffer(weight_tensor1, w_buf1, buf_size);

    release_graph_tensor(weight_tensor);
    release_graph_tensor(weight_tensor1);

    /* set bias */

    int input_num = get_node_input_number(conv_node);
    float* b_buf = nullptr;
    float* b_buf1 = nullptr;

    if(input_num > 2)
    {
        tensor_t bias_tensor = get_node_input_tensor(conv_node, 2);
        tensor_t bias_tensor1 = get_node_input_tensor(conv_node1, 2);

        buf_size = get_tensor_buffer_size(bias_tensor);
        b_buf = ( float* )malloc(buf_size);
        b_buf1 = ( float* )malloc(buf_size);

        for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
        {
            b_buf[i] = 1;
            b_buf1[i] = 1;
        }

        set_tensor_buffer(bias_tensor, b_buf, buf_size);
        set_tensor_buffer(bias_tensor1, b_buf1, buf_size);
        release_graph_tensor(bias_tensor);
        release_graph_tensor(bias_tensor1);
    }

    // dump_graph(graph);
    // prerun graph
    setenv("OPS_REGISTRY", "reference", 1);
    setenv("OP_NAME", "Convolution", 1);
    if(prerun_graph(graph1) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    unsetenv("OPS_REGISTRY");
    unsetenv("OP_NAME");
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    if(run_graph(graph, 1) < 0 || run_graph(graph1, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }

    tensor_t output_tensor = get_node_output_tensor(conv_node, 0);
    tensor_t output_tensor1 = get_node_output_tensor(conv_node1, 0);
    int size = get_tensor_buffer_size(output_tensor);
    int8_t* buf = ( int8_t* )get_tensor_buffer(output_tensor);
    int8_t* buf1 = ( int8_t* )get_tensor_buffer(output_tensor1);
    int ret = 0;

    if(int8_mismatch(buf, buf1, size) != 0)
        ret = 1;

    release_graph_tensor(output_tensor);
    release_graph_node(conv_node);

    postrun_graph(graph);

    destroy_graph(graph);

    free(i_buf);
    free(w_buf);
    free(i_buf1);
    free(w_buf1);

    if(b_buf)
        free(b_buf);
    if(b_buf1)
        free(b_buf1);
    return ret;
}

int main(int argc, char* argv[])
{
    init_tengine();
    int fail_num = 0;

    //       c   h   w   k  s  p_h  p_w   d  o  g
    /*============== test stride = 1 pad = 0 ===================*/
    fail_num += test_conv(4, 10, 10, 3, 1, 0, 0, 1, 4, 4);    // mid_h_loop_8
    fail_num += test_conv(4, 18, 18, 3, 1, 0, 0, 1, 4, 4);    // mid_h_loop_8
    fail_num += test_conv(4, 6, 6, 3, 1, 0, 0, 1, 4, 4);    // mid_h_loop_4
    fail_num += test_conv(4, 14, 14, 3, 1, 0, 0, 1, 4, 4);    // mid_h_loop_8 + mid_h_loop_4
    fail_num += test_conv(4, 6, 6, 3, 1, 0, 0, 1, 4, 4);    // mid_h_loop_2
    fail_num += test_conv(4, 6, 3, 3, 1, 0, 0, 1, 4, 4);    // last 3
    fail_num += test_conv(4, 6, 3, 3, 1, 0, 0, 1, 4, 4);    // last 2
    if(fail_num > 0)
    {
        release_tengine();
        printf("depth wise stride = 1 pad = 0 test failed\n");
        return -1;
    }
    fail_num = 0;

    /*============== test stride = 1 pad = 1 ===================*/
    fail_num += test_conv(4, 2, 9, 3, 1, 1, 1, 1, 4, 4);    //(first_2h last_2h)first_8 + last_2
    fail_num += test_conv(4, 2, 17, 3, 1, 1, 1, 1, 4, 4);    //(first_2h last_2h)first_8 + loop_8 + last_2
    fail_num += test_conv(4, 2, 5, 3, 1, 1, 1, 1, 4, 4);    //(first_2h last_2h)first_4 + last_2
    fail_num += test_conv(4, 2, 10, 3, 1, 1, 1, 1, 4, 4);    //(first_2h last_2h)first_4 + loop_4 + last_3
    fail_num += test_conv(4, 2, 3, 3, 1, 1, 1, 1, 4, 4);    //(first_2h last_2h)first_2 + last_2
    fail_num += test_conv(4, 4, 9, 3, 1, 1, 1, 1, 4, 4);    //(first_2h+ mid_2h+ last_2h)first_8 + last_2
    fail_num += test_conv(4, 4, 17, 3, 1, 1, 1, 1, 4, 4);    //(first_2h+ mid_2h+ last_2h)first_8 + loop_8 + last_2
    fail_num += test_conv(4, 4, 5, 3, 1, 1, 1, 1, 4, 4);    //(first_2h+ mid_2h+ last_2h)first_4 + last_2
    fail_num += test_conv(4, 4, 10, 3, 1, 1, 1, 1, 4, 4);    //(first_2h+ mid_2h+ last_2h)first_4 + loop_4 + last_3
    fail_num += test_conv(4, 4, 3, 3, 1, 1, 1, 1, 4, 4);    //(first_2h+ mid_2h+ last_2h)first_2 + last_2
    if(fail_num > 0)
    {
        release_tengine();
        printf("depth wise stride = 1 pad = 1 test failed\n");
        return -1;
    }
    fail_num = 0;
    /*============== test stride = 2 tensorflow VALID ===================*/
    fail_num += test_conv(4, 9, 9, 3, 2, 0, 0, 1, 4, 4);    // mid_h_loop_4
    fail_num += test_conv(4, 9, 5, 3, 2, 0, 0, 1, 4, 4);    // mid_h_loop_2
    if(fail_num > 0)
    {
        release_tengine();
        printf("depth wise stride = 2, pad = 0 test failed\n");
        return -1;
    }
    fail_num = 0;
    /*============== test stride = 2  pad = 1 ===================*/
    fail_num += test_conv(4, 2, 8, 3, 2, 1, 1, 1, 4, 4);    //(first2h last2h) first4 + last2
    fail_num += test_conv(4, 2, 4, 3, 2, 1, 1, 1, 4, 4);    //(first2h last2h) first2 + last1
    fail_num += test_conv(4, 8, 8, 3, 2, 1, 1, 1, 4, 4);    //(first2h + mid_h + last2h) first4 + last2
    fail_num += test_conv(4, 8, 4, 3, 2, 1, 1, 1, 4, 4);    //(first2h + mid_h + last2h) first2 + last1
    fail_num += test_conv(4, 8, 8, 3, 2, 1, 1, 1, 4, 4);    //(first2h + mid_h + last2h) first4 + last2
    fail_num += test_conv(4, 8, 4, 3, 2, 1, 1, 1, 4, 4);    //(first2h + mid_h + last2h) first2 + last1
    fail_num += test_conv(4, 8, 19, 3, 2, 1, 1, 1, 4, 4);    //(first2h + mid_h + last2h) first4 + loop4 + last4 + last2
    fail_num += test_conv(4, 8, 18, 3, 2, 1, 1, 1, 4, 4);    //(first2h + mid_h + last2h) first4 + loop4 + last3 + last1
    fail_num += test_conv(4, 8, 32, 3, 2, 1, 1, 1, 4, 4);    //(first2h + mid_h + last2h)
    if(fail_num > 0)
    {
        release_tengine();
        printf("depth wise stride = 2 pad = 1 test failed\n");
        return -1;
    }
    fail_num = 0;

    /*============== test stride = 2  tensorflow SAME ===================*/
    fail_num += test_conv(4, 2, 20, 3, 2, -1, -1, 1, 4, 4);
    fail_num += test_conv(4, 2, 10, 3, 2, -1, -1, 1, 4, 4);
    fail_num += test_conv(4, 5, 20, 3, 2, -1, -1, 1, 4, 4);
    fail_num += test_conv(4, 5, 10, 3, 2, -1, -1, 1, 4, 4);

    fail_num += test_conv(4, 10, 20, 3, 2, -1, -1, 1, 4, 4);
    fail_num += test_conv(4, 10, 10, 3, 2, -1, -1, 1, 4, 4);

    if(fail_num > 0)
    {
        release_tengine();
        printf("depth wise stride = 1 tensorflow same test failed\n");
        return -1;
    }
    fail_num = 0;
    /*============== test stride = 2  Onnx SAME ===================*/
    fail_num += test_conv(4, 2, 20, 3, 2, -2, -2, 1, 4, 4);
    fail_num += test_conv(4, 2, 10, 3, 2, -2, -2, 1, 4, 4);
    fail_num += test_conv(4, 7, 20, 3, 2, -2, -2, 1, 4, 4);
    fail_num += test_conv(4, 7, 10, 3, 2, -2, -2, 1, 4, 4);
    fail_num += test_conv(4, 10, 20, 3, 2, -1, -1, 1, 4, 4);
    fail_num += test_conv(4, 10, 10, 3, 2, -1, -1, 1, 4, 4);

    if(fail_num > 0)
    {
        release_tengine();
        printf("depth wise stride = 1 onnx SAME test failed\n");
        return -1;
    }
    fail_num = 0;
    release_tengine();

    printf("==============ALL DW Test PASS===============\n");

    return 0;
}
