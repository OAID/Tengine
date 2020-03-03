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

    int dims[4] = {1, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_input_node1(graph_t graph, const char* node_name, int data_type)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[1] = {1,};

    set_tensor_shape(tensor, dims, 1);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}


int create_test_node(graph_t graph, const char* node_name, const char* input_name, const char* input_name_1)
{
    node_t test_node = create_graph_node(graph, node_name, "Logical");

    const int type = 1;
    set_node_attr_int(test_node, "type", &type);
    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    tensor_t input_tensor_1 = get_graph_tensor(graph, input_name_1);
    // int data_type = get_tensor_data_type(input_tensor);
    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 0, input_tensor);
    set_node_input_tensor(test_node, 1, input_tensor_1);

    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor_1);

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
    const char* input_name_1 = "data1";


    if(create_input_node(graph, input_name, c, h, w, TENGINE_DT_FP32) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_input_node1(graph, input_name_1, TENGINE_DT_FP32) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }


    if(create_test_node(graph, test_node_name, input_name, input_name_1) < 0)
    {
        std::cerr << "create test node failed\n";
        return nullptr;
    }

    /* set input/output node */
    const char* inputs[] = {input_name, input_name_1};
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

std::vector<float*> set_input_data(graph_t graph)
{
    std::vector<float*> tmpbuf;
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);

    tensor_t input_tensor1 = get_graph_input_tensor(graph, 1, 0);

    int buf_size1 = get_tensor_buffer_size(input_tensor1);

    // input_tensor_1

    void* i_buf = malloc(buf_size);
    void* i_buf1 = malloc(buf_size1);

    float* tmp_b1= (float*) i_buf;
    float* tmp_b2= (float*) i_buf1;
    int dims[4];
    int dims1[1];

    get_tensor_shape(input_tensor, dims, 4);
    get_tensor_shape(input_tensor1, dims1, 1);

    int elem_num = dims[0] * dims[1] * dims[2] * dims[3];
    int elem_num1 = dims[0];
    int data_type = get_tensor_data_type(input_tensor);

    for(int i = 0; i < elem_num; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* f = ( float* )i_buf;
            f[i] = 1;
            // float* f1 = ( float* )i_buf1;
            // f1[i] = 0;
        }
        
    }

    for(int i = 0; i < elem_num1; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* f1 = ( float* )i_buf1;
            f1[i] = 0;
        }
        
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);

    release_graph_tensor(input_tensor);

    set_tensor_buffer(input_tensor1, i_buf1, buf_size1);

    release_graph_tensor(input_tensor1);
    tmpbuf.push_back(tmp_b1);
    tmpbuf.push_back(tmp_b2);
    return tmpbuf;
}

void dump_output_data(node_t test_node)
{
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    int dims[4];

    get_tensor_shape(output_tensor, dims, 4);

    void* o_buf = get_tensor_buffer(output_tensor);
    int data_type = get_tensor_data_type(output_tensor);
    

    int flag = 0;
    for(int i = 0; i < dims[1] * dims[0] * dims[2] * dims[3]; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* p = ( float* )o_buf;
            if(p[i] != 1){
	        flag = 1;
	    }
	    //std::cout << i << " " << p[i] << "\n";
        }
        
    }
    if(flag == 0){
        std::cout << "pass\n";
    }
    else{
    	std::cout << "test failed\n";
    }


    release_graph_tensor(output_tensor);
}

int main(int argc, char* argv[])
{
    int c = 3, h = 6, w = 7;
    const char* test_node_name = "logical";
    
    init_tengine();

    graph_t graph = create_test_graph(test_node_name, c, h, w);

    if(graph == nullptr)
        return 1;

    /* set input */
    std::vector<float*> tmp_i_buf;
    tmp_i_buf = set_input_data(graph);
    
    // tensor_t output = get_graph_output_tensor(graph, 0, 0);
    
    // prerun graph
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
    // dump_graph(graph);
    dump_output_data(test_node);

    release_graph_node(test_node);

    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();
    return 0;
}
