#include <unistd.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w, int data_type)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {3, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_input_node2(graph_t graph, const char* node_name, int data_type)
{
    node_t node = create_graph_node(graph, node_name, "Const");
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_CONST);

    int dims[1] = {1,};

    set_tensor_shape(tensor, dims, 1);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_test_node(graph_t graph, const char* node_name, const char* input_name, const char* input_name2)
{
    node_t test_node = create_graph_node(graph, node_name, "Reverse");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    tensor_t input_tensor2 = get_graph_tensor(graph, input_name2);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 0, input_tensor);
    set_node_input_tensor(test_node, 1, input_tensor2);
    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor2);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph(const char* test_node_name, int c, int h, int w)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);

    if(graph == nullptr)
    {
        std::cerr << "create failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_layout(graph, TENGINE_LAYOUT_NCHW) < 0)
    {
        std::cerr << "set layout failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* input_name2 = "axis";

    if(create_input_node(graph, input_name, c, h, w, TENGINE_DT_FP32) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_input_node2(graph, input_name2, TENGINE_DT_INT32) < 0)
    {
        std::cerr << "create input2 failed\n";
        return nullptr;
    }

    if(create_test_node(graph, test_node_name, input_name, input_name2) < 0)
    {
        std::cerr << "create test node failed\n";
        return nullptr;
    }

    /* set input/output node */
    const char* inputs[] = {input_name, input_name2};
    const char* outputs[] = {test_node_name};

    if(set_graph_input_node(graph, inputs, 2) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_output_node(graph, outputs, 1) < 0)
    {
        std::cerr << "set outputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    return graph;
}

void set_input_data(graph_t graph)
{
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t input_tensor2 = get_graph_input_tensor(graph, 1, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    int buf_size2 = get_tensor_buffer_size(input_tensor2);

    void* i_buf = malloc(buf_size);
    void* i_buf2 = malloc(buf_size2);
    int dims[4];
    int dims2[1];

    get_tensor_shape(input_tensor, dims, 4);
    get_tensor_shape(input_tensor2, dims2, 1);
    int data_type = get_tensor_data_type(input_tensor);
    int data_type2 = get_tensor_data_type(input_tensor2);

    int elem_num = dims[0] * dims[1] * dims[2] * dims[3];

    for(int i = 0; i < elem_num; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* f = ( float* )i_buf;
            f[i] = i;
        }
    }
    if(data_type2 == TENGINE_DT_INT32)
    {
        int* a = ( int* )i_buf2;
        a[0] = -4;
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    set_tensor_buffer(input_tensor2, i_buf2, buf_size2);

    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor2);
}

void dump_output_data(node_t test_node)
{
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    int dims[4];

    get_tensor_shape(output_tensor, dims, 4);

    void* o_buf = get_tensor_buffer(output_tensor);
    int data_type = get_tensor_data_type(output_tensor);

    for(int i = 0; i < dims[0]; i++)
    {
        for(int j = 0; j < dims[1] * dims[2] * dims[3]; j++)
        {
            if(data_type == TENGINE_DT_FP32)
            {
                float* p = ( float* )o_buf;
                if(i == 0)
                {
                    if(p[j] != 54 + j)
                    {
                        release_graph_tensor(output_tensor);
                        std::cout << "FAILED" << std::endl;
                        return;
                    }
                }

                if(i == 1)
                {
                    if(p[j + 27] != 27 + j)
                    {
                        release_graph_tensor(output_tensor);
                        std::cout << "FAILED" << std::endl;
                        return;
                    }
                }

                if(i == 2)
                {
                    if(p[j + 54] != 0 + j)
                    {
                        release_graph_tensor(output_tensor);
                        std::cout << "FAILED" << std::endl;
                        return;
                    }
                }
                
            }
        }
        
    }

    // for(int i = 0; i < dims[0] * dims[1] * dims[2] * dims[3]; i++)
    // {
    //     if(data_type == TENGINE_DT_FP32)
    //     {
    //         float* p = ( float* )o_buf;
    //         std::cout << p[i] << std::endl;
    //     }
        
    // }

    std::cout << "pass" << std::endl;
    release_graph_tensor(output_tensor);
    return;
}

int main(int argc, char* argv[])
{
    int c = 3, h = 3, w = 3;
    const char* test_node_name = "reverse";

    init_tengine();

    graph_t graph = create_test_graph(test_node_name, c, h, w);

    if(graph == nullptr)
        return 1;

    /* set input */
    set_input_data(graph);

    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    dump_graph(graph);

    node_t test_node = get_graph_node(graph, test_node_name);

    const char* dev = get_node_device(test_node);

    std::cout << "node running on dev: " << dev << "\n";

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }

    dump_output_data(test_node);

    release_graph_node(test_node);

    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();
    return 0;
}