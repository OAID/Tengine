#include <unistd.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

int create_input_node1(graph_t graph)
{
    node_t node = create_graph_node(graph, "input1", "InputOp");
    tensor_t tensor = create_graph_tensor(graph, "input1", TENGINE_DT_FP32);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, 4, 4, 3};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_input_node2(graph_t graph)
{
    node_t node = create_graph_node(graph, "input2", "InputOp");
    tensor_t tensor = create_graph_tensor(graph, "input2", TENGINE_DT_FP32);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, 4, 4, 3};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_test_node(graph_t graph)
{
    node_t test_node = create_graph_node(graph, "SquaredDifference", "SquaredDifference");

    tensor_t input_tensor_1= get_graph_tensor(graph, "input1");
    tensor_t input_tensor_2 = get_graph_tensor(graph, "input2");

    if((input_tensor_1 == nullptr) | (input_tensor_2 == nullptr))
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 0, input_tensor_1);
    set_node_input_tensor(test_node, 1, input_tensor_2);

    release_graph_tensor(input_tensor_1);
    release_graph_tensor(input_tensor_2);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, "SquaredDifference", TENGINE_DT_FP32);

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

    if(set_graph_layout(graph, TENGINE_LAYOUT_NHWC) < 0)
    {
        std::cerr << "set layout failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(create_input_node1(graph) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_input_node2(graph) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }


    if(create_test_node(graph) < 0)
    {
        std::cerr << "create test node failed\n";
        return nullptr;
    }

    /* set input/output node */
    const char* inputs[] = {"input1", "input2"};
    const char* outputs[] = {"SquaredDifference"};

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
    tensor_t input_tensor1 = get_graph_input_tensor(graph, 0, 0);

    int buf_size1 = get_tensor_buffer_size(input_tensor1);

    tensor_t input_tensor2 = get_graph_input_tensor(graph, 1, 0);

    int buf_size2 = get_tensor_buffer_size(input_tensor2);

    // input_tensor_1

    void* i_buf1 = malloc(buf_size1);
    void* i_buf2 = malloc(buf_size2);

    float* tmp_b1= (float*) i_buf1;
    float* tmp_b2= (float*) i_buf2;
    int dims1[4];
    int dims2[4];

    get_tensor_shape(input_tensor1, dims1, 4);
    get_tensor_shape(input_tensor2, dims2, 4);

    int elem_num1 = dims1[0] * dims1[1] * dims1[2] * dims1[3];
    int elem_num2 = dims2[0] * dims2[1] * dims2[2] * dims2[3];
    int data_type = get_tensor_data_type(input_tensor1);

    for(int i = 0; i < elem_num1; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* f = ( float* )i_buf1;
            f[i] = 9;
        }
        
    }

    for(int i = 0; i < elem_num2; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* f1 = ( float* )i_buf2;
            f1[i] = 5;
        }
        
    }

    set_tensor_buffer(input_tensor1, i_buf1, buf_size1);

    release_graph_tensor(input_tensor1);

    set_tensor_buffer(input_tensor2, i_buf2, buf_size2);

    release_graph_tensor(input_tensor2);
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
            if(p[i] != 16){
	        flag = 1;
	    }
	    // std::cout << " " << p[i] << "\n";
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
    init_tengine();

    graph_t graph = create_test_graph();

    if(graph == nullptr)
        return 1;

    /* set input */
    std::vector<float*> tmp_i_buf;
    tmp_i_buf = set_input_data(graph);

    // prerun graph
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    dump_graph(graph);
    node_t test_node = get_graph_node(graph, "SquaredDifference");

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