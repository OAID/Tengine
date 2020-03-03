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

    const char* model_file = "./models/test_sparsetodense.pb";
    graph_t graph = create_graph(nullptr, "tensorflow", model_file);
    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    prerun_graph(graph);
    dump_graph(graph);
    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 1, 0);
    if(NULL == output_tensor)
    {
        std::cout << "output tensor is NULL\n";
    }

    int dims[1];
    int dim_size = get_tensor_shape(output_tensor, dims, 1);

    if(dim_size < 0)
    {
        printf("get output tensor shape failed\n");
        return -1;
    }

    float* data = ( float* )(get_tensor_buffer(output_tensor));

    int a = 0;
    for(int i = 0; i < dims[0]; i++)
    {
        float* p = ( float* )data;
        // std::cout << p[i] << std::endl;
        if(i < 3){
            if(p[i] != 6 - i){
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
    if(a == 0){
        printf("pass\n");
    }
    else
    {
        printf("FAIL\n");
    }
    
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}