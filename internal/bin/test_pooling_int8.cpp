#include <iostream>
#include <string>

/* the sample code to create a poololution and do calculation */

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
        printf("mismatch:\n\t[%d]\t---a:    %d ,b:  :%d---        off: %d\n", i, a[i], b[i], a[i] - b[i]);
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

int create_pool_node(graph_t graph, const char* node_name, const char* input_name, int kernel_h, int kernel_w,
                     int stride_h, int stride_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int method)
{
    node_t pool_node = create_graph_node(graph, node_name, "Pooling");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(pool_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_INT8);
    set_node_output_tensor(pool_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* attr */
    set_node_attr_int(pool_node, "kernel_h", &kernel_h);
    set_node_attr_int(pool_node, "kernel_w", &kernel_w);
    set_node_attr_int(pool_node, "stride_h", &stride_h);
    set_node_attr_int(pool_node, "stride_w", &stride_w);
    set_node_attr_int(pool_node, "pad_h0", &pad_h0);
    set_node_attr_int(pool_node, "pad_w0", &pad_w0);
    set_node_attr_int(pool_node, "pad_h1", &pad_h1);
    set_node_attr_int(pool_node, "pad_w1", &pad_w1);
    set_node_attr_int(pool_node, "alg", &method);

    release_graph_node(pool_node);

    return 0;
}

graph_t create_pool_graph(int c, int h, int w, int k_h, int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1,
                          int pad_w1, int m)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* pool_name = "pool";

    if(create_input_node(graph, input_name, c, h, w) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_pool_node(graph, pool_name, input_name, k_h, k_w, s_h, s_w, pad_h0, pad_w0, pad_h1, pad_w1, m) < 0)
    {
        std::cerr << "create pool node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {pool_name};

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

int test_pool(int c, int h, int w, int k_h, int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1,
              int m)
{
    graph_t graph = create_pool_graph(c, h, w, k_h, k_w, s_h, s_w, pad_h0, pad_w0, pad_h1, pad_w1, m);
    graph_t graph1 = create_pool_graph(c, h, w, k_h, k_w, s_h, s_w, pad_h0, pad_w0, pad_h1, pad_w1, m);
    if(graph == nullptr || graph1 == nullptr)
        return 1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t input_tensor1 = get_graph_input_tensor(graph1, 0, 0);

    float scale = 0.1;
    int zero = 0;
    set_tensor_quant_param(input_tensor, &scale, &zero, 1);
    set_tensor_quant_param(input_tensor1, &scale, &zero, 1);

    int buf_size = get_tensor_buffer_size(input_tensor);
    int8_t* i_buf = ( int8_t* )malloc(buf_size);
    int8_t* i_buf1 = ( int8_t* )malloc(buf_size);

    for(int i = 0; i < buf_size; i++)
    {
        i_buf[i] = 1;    // random()/127;
        i_buf1[i] = i_buf[i];
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    set_tensor_buffer(input_tensor1, i_buf1, buf_size);
    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor1);

    // dump_graph(graph);
    // prerun graph
    setenv("OPS_REGISTRY", "reference", 1);
    setenv("OP_NAME", "Pooling", 1);
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

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    tensor_t output_tensor1 = get_graph_output_tensor(graph1, 0, 0);
    int size = get_tensor_buffer_size(output_tensor);
    int8_t* buf = ( int8_t* )get_tensor_buffer(output_tensor);
    int8_t* buf1 = ( int8_t* )get_tensor_buffer(output_tensor1);

    int ret = 0;
    if(int8_mismatch(buf, buf1, size) != 0)
    {
        ret = -1;
        printf(" not match \n-------------------------------------------------\n");
    }

    release_graph_tensor(output_tensor);
    release_graph_tensor(output_tensor1);

    postrun_graph(graph);

    destroy_graph(graph);

    free(i_buf);
    free(i_buf1);

    return ret;
}

int main(int argc, char* argv[])
{
    init_tengine();

    //       c   h   w   k_h   k_w   s_h   s_w  p_h0  p_w0  p_h1  p_w1  m
    /*
     */
    // test max pooling 2x2 _s2
    int failed_num = 0;

    failed_num += test_pool(1, 56, 56, 2, 2, 2, 2, 0, 0, 0, 0, 0);
    failed_num += test_pool(6, 56, 56, 2, 2, 2, 2, 0, 0, 0, 0, 0);
    failed_num += test_pool(5, 3, 56, 2, 2, 2, 2, 0, 0, 0, 0, 0);
    failed_num += test_pool(3, 3, 12, 2, 2, 2, 2, 0, 0, 0, 0, 0);
    failed_num += test_pool(6, 54, 54, 2, 2, 2, 2, 0, 0, 1, 0, 0);
    failed_num += test_pool(6, 56, 56, 2, 2, 2, 2, 0, 0, 0, 1, 0);
    failed_num += test_pool(6, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, 0);
    failed_num += test_pool(6, 23, 23, 2, 2, 2, 2, 0, 0, 1, 1, 0);
    if(failed_num < 0)
    {
        printf("max pooling 2x2 test failed\n");
        release_tengine();
        return 0;
    }
    // test avg pooling 2x2 s2
    failed_num = 0;

    failed_num += test_pool(1, 56, 56, 2, 2, 2, 2, 0, 0, 0, 0, 1);
    failed_num += test_pool(6, 56, 56, 2, 2, 2, 2, 0, 0, 0, 0, 1);
    failed_num += test_pool(5, 3, 56, 2, 2, 2, 2, 0, 0, 0, 0, 1);
    failed_num += test_pool(3, 3, 12, 2, 2, 2, 2, 0, 0, 0, 0, 1);
    failed_num += test_pool(6, 54, 54, 2, 2, 2, 2, 0, 0, 1, 0, 1);
    failed_num += test_pool(6, 56, 56, 2, 2, 2, 2, 0, 0, 0, 1, 1);
    failed_num += test_pool(6, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, 1);
    failed_num += test_pool(6, 23, 23, 2, 2, 2, 2, 0, 0, 1, 1, 1);
    if(failed_num < 0)
    {
        printf("average pooling 2x2_s2 test failed\n");
        release_tengine();
        return 0;
    }
    // test max pooling 3x3 s2
    failed_num = 0;

    failed_num += test_pool(1, 56, 56, 3, 3, 2, 2, 0, 0, 0, 0, 0);
    failed_num += test_pool(6, 56, 56, 3, 3, 2, 2, 0, 0, 0, 0, 0);
    failed_num += test_pool(5, 3, 56, 3, 3, 2, 2, 0, 0, 0, 0, 0);
    failed_num += test_pool(3, 3, 12, 3, 3, 2, 2, 0, 0, 0, 0, 0);
    failed_num += test_pool(5, 4, 56, 3, 3, 2, 2, 1, 1, 1, 1, 0);
    failed_num += test_pool(5, 12, 51, 3, 3, 2, 2, 1, 1, 1, 1, 0);
    failed_num += test_pool(5, 4, 56, 3, 3, 2, 2, 0, 0, 1, 1, 0);
    failed_num += test_pool(3, 5, 41, 3, 3, 2, 2, 0, 0, 1, 1, 0);
    failed_num += test_pool(5, 4, 56, 3, 3, 2, 2, 0, 0, 1, 0, 0);
    failed_num += test_pool(5, 4, 56, 3, 3, 2, 2, 0, 0, 0, 1, 0);

    if(failed_num < 0)
    {
        printf("max pooling 3x3_s2 test failed\n");
        release_tengine();
        return 0;
    }

    // test avg pooling 3x3 s2
    failed_num += test_pool(1, 56, 56, 3, 3, 2, 2, 0, 0, 0, 0, 1);
    failed_num += test_pool(6, 56, 56, 3, 3, 2, 2, 0, 0, 0, 0, 1);
    failed_num += test_pool(5, 3, 56, 3, 3, 2, 2, 0, 0, 0, 0, 1);
    failed_num += test_pool(3, 3, 12, 3, 3, 2, 2, 0, 0, 0, 0, 1);
    failed_num += test_pool(5, 4, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1);
    failed_num += test_pool(5, 12, 51, 3, 3, 2, 2, 1, 1, 1, 1, 1);
    failed_num += test_pool(5, 4, 56, 3, 3, 2, 2, 0, 0, 1, 1, 1);
    failed_num += test_pool(3, 5, 41, 3, 3, 2, 2, 0, 0, 1, 1, 1);
    failed_num += test_pool(5, 4, 56, 3, 3, 2, 2, 0, 0, 1, 0, 1);
    failed_num += test_pool(5, 4, 56, 3, 3, 2, 2, 0, 0, 0, 1, 1);
    if(failed_num < 0)
    {
        printf("average pooling 3x3_s2 test failed\n");
        release_tengine();
        return 0;
    }

    // test max pooling 3x3 s1
    failed_num += test_pool(1, 56, 56, 3, 3, 1, 1, 0, 0, 0, 0, 0);
    failed_num += test_pool(6, 56, 56, 3, 3, 1, 1, 0, 0, 0, 0, 0);
    failed_num += test_pool(5, 3, 56, 3, 3, 1, 1, 0, 0, 0, 0, 0);
    failed_num += test_pool(3, 3, 12, 3, 3, 1, 1, 0, 0, 0, 0, 0);
    failed_num += test_pool(5, 4, 56, 3, 3, 1, 1, 1, 1, 1, 1, 0);
    failed_num += test_pool(5, 12, 51, 3, 3, 1, 1, 1, 1, 1, 1, 0);
    failed_num += test_pool(5, 4, 56, 3, 3, 1, 1, 0, 0, 1, 1, 0);
    failed_num += test_pool(5, 4, 56, 3, 3, 1, 1, 0, 0, 0, 1, 0);
    failed_num += test_pool(5, 4, 56, 3, 3, 1, 1, 0, 0, 1, 0, 0);
    if(failed_num < 0)
    {
        printf("max pooling 3x3_s1 test failed\n");
        release_tengine();
        return 0;
    }

    // test avg pooling 3x3 s1
    failed_num += test_pool(1, 56, 56, 3, 3, 1, 1, 0, 0, 0, 0, 1);
    failed_num += test_pool(6, 56, 56, 3, 3, 1, 1, 0, 0, 0, 0, 1);
    failed_num += test_pool(5, 3, 56, 3, 3, 1, 1, 0, 0, 0, 0, 1);
    failed_num += test_pool(3, 3, 12, 3, 3, 1, 1, 0, 0, 0, 0, 1);
    failed_num += test_pool(5, 4, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1);
    failed_num += test_pool(5, 12, 51, 3, 3, 1, 1, 1, 1, 1, 1, 1);
    failed_num += test_pool(5, 4, 56, 3, 3, 1, 1, 0, 0, 1, 1, 1);
    failed_num += test_pool(5, 4, 56, 3, 3, 1, 1, 0, 0, 0, 1, 1);
    failed_num += test_pool(5, 4, 56, 3, 3, 1, 1, 0, 0, 1, 0, 1);
    if(failed_num < 0)
    {
        release_tengine();
        printf("average pooling 3x3_s1 test failed\n");
        return 0;
    }

    release_tengine();
    printf("===================ALL Test Pass!==================\n");
    return 0;
}
