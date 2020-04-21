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

const char* model_name = "../models/instancenorm.tmfile";

const std::vector<float> test_data1 = {0.11, 0.23, 0.14, 0.15,
                                       0.24, 0.56, 0.33, 0.22,
                                       0.47, 0.74, 0.26, 0.42,
                                       0.64, 0.79, 0.23, 0.79,
                                       0.16, 0.93, 0.34, 0.65,
                                       0.74, 0.76, 0.43, 0.82,
                                       0.67, 0.94, 0.76, 0.62,
                                       0.24, 0.49, 0.33, 0.69,
                                       0.51, 0.83, 0.44, 0.75,
                                       0.44, 0.56, 0.53, 0.62,
                                       0.37, 0.78, 0.56, 0.45,
                                       0.64, 0.79, 0.78, 0.67};

float result[48] = {
-1.2274680, -0.7106394, -1.0982609, -1.0551919,
-0.6675705, 0.7106392, -0.2799488, -0.7537085,
0.3230178, 1.4858823, -0.5814323, 0.1076725,
1.0551918, 1.7012277, -0.7106394, 1.7012277,
-1.6216202, -0.1228682, -1.2712625, -0.6678690,
-0.4926901, -0.4537615, -1.0960836, -0.3369757,
-0.6289402, -0.1034039, -0.4537615, -0.7262619,
-1.4659057, -0.9792979, -1.2907269, -0.5900117,
-0.9675574, -0.6836667, -1.0296586, -0.7546394,
-1.0296586, -0.9231995, -0.9498143, -0.8699700,
-1.0917597, -0.7280247, -0.9231995, -1.0207870,
-0.8522269, -0.7191530, -0.7280247, -0.8256121

};

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
    graph_t graph = create_graph(nullptr, "tengine", model_name);
    //graph_t graph = create_graph(nullptr, "mxnet", model_name1, model_name2);

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

    int h = 4;
    int w = 4;
    int flatten_size = 3 * h * w;

    int dims[] = {1,3,h, w};
    set_tensor_shape(input_tensor, dims, 4);
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
        //print_output(output, flatten_size);
        for(int i = 0; i < flatten_size; i++){
            float tmp = fabs(output[i]) - fabs(result[i]);
            if(tmp > 0.00001){
                printf("Test Failed \n");
                return 0;
            }           
        }
    }
    printf("pass\n");
    if(postrun_graph(graph) != 0)
    {
        std::cout << "postrun graph failed\n";
    }
    free(input_data);

    destroy_graph(graph);
    printf("All Test Done\n");
    return 0;
    
}

