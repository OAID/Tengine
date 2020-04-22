#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include <math.h>

const char* model_name = "../models/embed.tmfile";

const std::vector<float> test_data1 = {1, 3, 0, 2};
float result[20] ={5.0000000, 6.0000000, 7.0000000, 8.0000000,
                 9.0000000, 15.0000000, 16.0000000, 17.0000000,
                 18.0000000, 19.0000000, 0.0000000, 1.0000000, 
                 2.0000000, 3.0000000, 4.0000000, 10.0000000, 
                 11.0000000, 12.0000000, 13.0000000, 14.0000000};

void set_input_data(float* input_data, int num, int size)
{
    float* tmp = input_data;
    switch (num)
    {
    case 1:
        for(int i = 0; i < size; i++)
            *tmp++ = test_data1[i];
        break;
    default:
        break;
    }
}

int main(int argc, char* argv[])
{
    if(init_tengine() < 0)
    {
        std::cout<< "init tengine failed" <<std::endl;
        return 1;
    }
        
    //std::cout<< "init tengine done." <<std::endl;
    //graph_t graph = create_graph(nullptr, "mxnet", model_name1, model_name2);
    graph_t graph = create_graph(nullptr, "tengine", model_name);

    if(graph == nullptr)
    {
        std::cout << "create graph failed!"<<std::endl;
        return 1;
    }
    //std::cout << "create graph done!"<<std::endl;

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    //if(!check_tensor_valid(input_tensor))
    //{
    //    printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
    //    return 1;
    //}

    int h = 2;
    int w = 2;
    int flatten_size = h * w;

    int dims[] = {h, w};
    set_tensor_shape(input_tensor, dims, 2);
    if(prerun_graph(graph) != 0)
    {
        std::cout << "prerun _graph failed\n";
        return -1;
    }
    //dump_graph(graph);
    float* input_data = (float*)malloc(sizeof(float)*flatten_size);
    for(int i = 1; i <= 1; i++)
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
        for(int i = 0; i < flatten_size; i++){
            float tmp = fabs(output[i]) - fabs(result[i]);
            if(tmp > 0.00001){
                printf("Test Failed \n");
                return 0;
            }           
        }
    }
    printf("pass\n");
    printf("All Test Done\n");
    if(postrun_graph(graph) != 0)
    {
        std::cout << "postrun graph failed\n";
    }
    free(input_data);

    destroy_graph(graph);

    return 0;
    
}

