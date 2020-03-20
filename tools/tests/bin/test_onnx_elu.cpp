#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include "tengine_operations.h"
#include "tengine_c_api.h"

const char* model_path = "./onnx-model/elu.onnx";
const std::vector<float> test_data1 = {-1.1, -0.6, -0.5};

const std::vector<float> test_data2 = {-1.9, -0.8, -0.4};

const std::vector<float> test_data3 = {1.1, 0.5, 0.2};

void set_input_data(float* input_data, int num, int size)
{
    float* tmp=input_data;
    switch (num)
    {
    case 1:
        for(int i = 0; i < size; i++)
            *tmp++ = test_data1[i];
        break;
    case 2:
        for(int i = 0; i < size; i++)
            *tmp++ = test_data2[i];
        break;
    case 3:
        for(int i = 0; i < size; i++)
            *tmp++ = test_data3[i];
        break;
    default:
        break;
    }
}

void print_output(float* output_data, int size)
{
    for(int i = 0; i < size; i++)
    {
        printf("%.7f ", *(output_data++));
    }
    printf("\n");
    //std::cout<<std::endl;
}

int main(int argc, char* argv[])
{
    if(init_tengine() < 0)
    {
        std::cout<< "init tengine failed" <<std::endl;
        return 1;
    }
        
    std::cout<< "init tengine done." <<std::endl;
    graph_t graph = create_graph(0, "onnx", model_path);
    if(graph == nullptr)
    {
        std::cout << "create graph failed!"<<std::endl;
        return 1;
    }
    std::cout << "create graph done!"<<std::endl;

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    //if(!check_tensor_valid(input_tensor))
    //{
    //    printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
    //    return 1;
    //}

    int w = 3;
    int flatten_size = w * 1;
    dump_graph(graph);
    int dims[] = {w};
    set_tensor_shape(input_tensor, dims, 1);
    if(prerun_graph(graph) != 0)
    {
        std::cout << "prerun _graph failed\n";
        return -1;
    }
    float* input_data = (float*)malloc(sizeof(float)*flatten_size);
    for(int i = 1; i <= 3; i++)
    {
        
        set_input_data(input_data, i, flatten_size);
        set_tensor_buffer(input_tensor, input_data, flatten_size);
        if(run_graph(graph, 1) != 0)
        {
            std::cout << "run _graph failed\n";
            return -1;
        }

        tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
        float* output = ( float* )(get_tensor_buffer(output_tensor));
        print_output(output, flatten_size);
    }

    if(postrun_graph(graph) != 0)
    {
        std::cout << "postrun graph failed\n";
    }
    free(input_data);

    destroy_graph(graph);

    return 0;
    
}

