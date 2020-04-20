#include <unistd.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

int create_indices_node(graph_t graph, int data_type)
{
    node_t node = create_graph_node(graph, "indices", "Const");
    tensor_t tensor = create_graph_tensor(graph, "indices", data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_CONST);

    int dims[2] = {3, 2};

    set_tensor_shape(tensor, dims, 2);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_output_shape_node(graph_t graph, int data_type)
{
    node_t node = create_graph_node(graph, "output_shape", "Const");
    tensor_t tensor = create_graph_tensor(graph, "output_shape", data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_CONST);

    int dims[2] = {1, 1};

    set_tensor_shape(tensor, dims, 2);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_sparse_value_node(graph_t graph, int data_type)
{
    node_t node = create_graph_node(graph, "sparse_value", "Const");
    tensor_t tensor = create_graph_tensor(graph, "sparse_value", data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_CONST);

    int dims[1] = {3};

    set_tensor_shape(tensor, dims, 1);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_test_node(graph_t graph)
{
    node_t test_node = create_graph_node(graph, "SparseToDense", "SparseToDense");
    const int default_value = 9;
    const int output_shape_size0 = 4;
    const int output_shape_size1 = 2;
    set_node_attr_int(test_node, "output_shape_size0", &output_shape_size0);
    set_node_attr_int(test_node, "default_value", &default_value);
    set_node_attr_int(test_node, "output_shape_size1", &output_shape_size1);

    tensor_t input_tensor = get_graph_tensor(graph, "indices");
    tensor_t input_tensor2 = get_graph_tensor(graph, "output_shape");
    tensor_t input_tensor3 = get_graph_tensor(graph, "sparse_value");

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 0, input_tensor);
    set_node_input_tensor(test_node, 1, input_tensor2);
    set_node_input_tensor(test_node, 2, input_tensor3);
    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor2);
    release_graph_tensor(input_tensor3);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, "SparseToDense", TENGINE_DT_FP32);

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph()
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

    if(create_indices_node(graph, TENGINE_DT_FP32) < 0)
    {
        return nullptr;
    }

    if(create_output_shape_node(graph, TENGINE_DT_INT32) < 0)
    {
        return nullptr;
    }

    if(create_sparse_value_node(graph, TENGINE_DT_INT32) < 0)
    {
        return nullptr;
    }

    if(create_test_node(graph) < 0)
    {
        return nullptr;
    }

    /* set input/output node */
    const char* inputs[] = {"indices", "output_shape", "sparse_value"};
    const char* outputs[] = {"SparseToDense"};

    if(set_graph_input_node(graph, inputs, 3) < 0)
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
    tensor_t indices_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t output_shape_tensor = get_graph_input_tensor(graph, 1, 0);
    tensor_t sparse_value_tensor = get_graph_input_tensor(graph, 2, 0);

    int buf_size = get_tensor_buffer_size(indices_tensor);
    int buf_size2 = get_tensor_buffer_size(output_shape_tensor);
    int buf_size3 = get_tensor_buffer_size(sparse_value_tensor);

    void* i_buf = malloc(buf_size);
    void* i_buf2 = malloc(buf_size2);
    void* i_buf3 = malloc(buf_size3);
    int dims[2];
    int dims2[2];
    int dims3[1];

    get_tensor_shape(indices_tensor, dims, 2);
    get_tensor_shape(output_shape_tensor, dims2, 2);
    get_tensor_shape(sparse_value_tensor, dims3, 1);
    // int data_type = get_tensor_data_type(indices_tensor);
    int data_type2 = get_tensor_data_type(output_shape_tensor);
    int data_type3 = get_tensor_data_type(sparse_value_tensor);

    // for(int i = 0; i < dims[0]; i++)
    // {
    //     if(data_type == TENGINE_DT_FP32)
    //     {
    //         int* f = ( int* )i_buf;
    //         f[i] = i;
    //     }
    // }
    int* f = ( int* )i_buf;
    f[0] = 0;
    f[1] = 0;
    f[2] = 0;
    f[3] = 1;
    f[4] = 1;
    f[5] = 0;

    if(data_type2 == TENGINE_DT_INT32)
    {
        int* a = ( int* )i_buf2;
        a[0] = 4;
        a[1] = 2;
    }
    for(int i = 0; i < dims3[0]; i++)
    {
        if(data_type3 == TENGINE_DT_INT32)
        {
            int* a = ( int* )i_buf3;
            a[i] = i - 6;
        }
    }

    set_tensor_buffer(indices_tensor, i_buf, buf_size);
    set_tensor_buffer(output_shape_tensor, i_buf2, buf_size2);
    set_tensor_buffer(sparse_value_tensor, i_buf3, buf_size3);

    release_graph_tensor(indices_tensor);
    release_graph_tensor(output_shape_tensor);
    release_graph_tensor(sparse_value_tensor);
}

void dump_output_data(node_t test_node)
{
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    int dims[2];

    get_tensor_shape(output_tensor, dims, 2);

    void* o_buf = get_tensor_buffer(output_tensor);
    int data_type = get_tensor_data_type(output_tensor);

    int a = 0;
    for(int i = 0; i < dims[0] * dims[1]; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* p = ( float* )o_buf;
            // std::cout << p[i] << std::endl;

            if(i < 3){
                if(p[i] != i - 6){
                    a = 1;
                }
            }
            else if (i >= 3)
            {
                if(p[i] != 9){
                    a = 1;
                }
            }
        }
    }
    if(a == 0){
        printf("pass\n");
    }
    else
    {
        printf("FAIL\n");
    }
    
    release_graph_tensor(output_tensor);
    return;
}

int main(int argc, char* argv[])
{
    init_tengine();

    graph_t graph = create_test_graph();

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

    node_t test_node = get_graph_node(graph, "SparseToDense");

    const char* dev = get_node_device(test_node);

    std::cout << "node running on dev: " << dev << "\n";

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    dump_output_data(test_node);

    release_graph_node(test_node);

    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();
    return 0;
}