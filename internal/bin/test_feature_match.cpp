#include <sys/time.h>
#include <iostream>
#include <string>

/* the sample code to create a fmolution and do calculation */

#include "tengine_c_api.h"

int create_input_node(graph_t graph, const char* node_name, int n, int k)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[2] = {n, k};

    set_tensor_shape(tensor, dims, sizeof(dims) / sizeof(int));

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_fm_node(graph_t graph, const char* node_name, const char* input_name, int num_output, int hidden_num)
{
    node_t fm_node = create_graph_node(graph, node_name, "FeatureMatch");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(fm_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(fm_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* weight */

    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    set_node_input_tensor(fm_node, 1, w_tensor);
    int w_dims[] = {num_output, hidden_num};

    set_tensor_shape(w_tensor, w_dims, 2);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    /* attr */
    set_node_attr_int(fm_node, "num_output", &num_output);

    release_graph_node(fm_node);

    return 0;
}

graph_t create_fm_graph(int n, int output_num, int hidden_num)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* fm_name = "fm";

    if(create_input_node(graph, input_name, n, hidden_num) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_fm_node(graph, fm_name, input_name, output_num, hidden_num) < 0)
    {
        std::cerr << "create fm node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {fm_name};

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

int main(int argc, char* argv[])
{
    int n, output_num, hidden_num;

    n = 1;
    output_num = 10;
    hidden_num = 20;

    init_tengine();

    graph_t graph = create_fm_graph(n, output_num, hidden_num);

    if(graph == nullptr)
        return 1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
        i_buf[i] = 1;    //(i%100 + 1)/100.0;

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);

    /* set weight */
    node_t fm_node = get_graph_node(graph, "fm");

    tensor_t weight_tensor = get_node_input_tensor(fm_node, 1);

    buf_size = get_tensor_buffer_size(weight_tensor);
    float* w_buf = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
        w_buf[i] = 1;    //(i%9+1)/100.0;

    set_tensor_buffer(weight_tensor, w_buf, buf_size);

    // prerun graph
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    int repeat_count = 1;

    struct timeval start;
    struct timeval end;

    gettimeofday(&start, NULL);

    for(int i = 0; i < repeat_count; i++)
    {
        if(run_graph(graph, 1) < 0)
        {
            std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
            return 2;
        }
    }

    gettimeofday(&end, NULL);

    long off = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);

    printf("total time: %ld repeate %d per run %.3f ms\n", off, repeat_count, 1.0 * off / repeat_count / 1000);

    tensor_t output_tensor = get_node_output_tensor(fm_node, 0);

    /* change the weight shape now */

    output_num = 20;

    int weight_dims[] = {output_num, hidden_num};

    set_tensor_shape(weight_tensor, weight_dims, sizeof(weight_dims) / sizeof(int));

    free(w_buf);

    w_buf = ( float* )malloc(output_num * hidden_num * sizeof(float));

    for(int i = 0; i < output_num * hidden_num; i++)
        w_buf[i] = 2;

    set_tensor_buffer(weight_tensor, w_buf, output_num * hidden_num * sizeof(float));

    int out_dims[] = {n, output_num};
    set_tensor_shape(output_tensor, out_dims, sizeof(out_dims) / sizeof(int));

    float* o_buf = ( float* )malloc(n * output_num * sizeof(float));

    set_tensor_buffer(output_tensor, o_buf, n * output_num * sizeof(float));

    int refreshed = 1;

    set_node_attr_int(fm_node, "refreshed", &refreshed);

    run_graph(graph, 1);

    float* buf = ( float* )get_tensor_buffer(output_tensor);

    for(int i = 0; i < n; i++)
        for(int j = 0; j < output_num; j++)
        {
            std::cout << " " << buf[i * output_num + j];
            if(j == output_num - 1)
                std::cout << "\n";
        }

    release_graph_tensor(weight_tensor);
    release_graph_tensor(output_tensor);
    release_graph_node(fm_node);

    postrun_graph(graph);

    destroy_graph(graph);

    free(i_buf);
    free(w_buf);
    free(o_buf);

    release_tengine();
    return 0;
}
