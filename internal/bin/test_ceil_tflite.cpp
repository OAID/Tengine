#include <unistd.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "tengine_config.hpp"

int main(int argc, char* argv[])
{
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    const char* model_file = "./models/test_ceil.tflite";
    graph_t graph = create_graph(nullptr, "tflite", model_file);
    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    tensor_t input_tensor1 = get_graph_input_tensor(graph, 0, 0);
    if(input_tensor1 == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d \n", 0, 0);
        return -1;
    }

    int dims[] = {1, 3, 3, 3};
    set_tensor_shape(input_tensor1, dims, 4);

    float* input_data1 = ( float* )malloc(sizeof(float) * 3 * 3 * 3 * 1) ;
    for(int i =0; i < 3 * 3 * 3; i ++)
    {
        input_data1[i] = i + 0.2;
    }

    if(set_tensor_buffer(input_tensor1, input_data1, 3 * 3 * 3 * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
    }

    prerun_graph(graph);
    dump_graph(graph);
    run_graph(graph, 1);
    
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    if(NULL == output_tensor)
    {
        std::cout << "output tensor is NULL\n";
    }

    int dims0[4];
    int dim_size = get_tensor_shape(output_tensor, dims0, 4);

    if(dim_size < 0)
    {
        printf("get output tensor shape failed\n");
        return -1;
    }

    int* data = ( int* )(get_tensor_buffer(output_tensor));

    int a = 0;
    for(int i = 0; i < dims0[0] * dims0[1] * dims0[2]; i++)
    {
        int* p = ( int* )data;
        // std::cout << p[i] << std::endl;
        if(i < 3){
            if(p[i] != i + 1){
                a = 1;
            }
        }

    }
    if(a == 0){
        printf("pass\n");
    }
    else
    {
        printf("fail\n");
    }

    release_graph_tensor(output_tensor);
    release_graph_tensor(input_tensor1);
    free(input_data1);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
