#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <math.h>

#include "tengine_c_api.h"
#include "compiler_fp16.h"
#include "operator/layernormlstm_param.hpp"

const char* inputTensorNames[24] = {"input",
                                  "i2i_weights", "i2c_weights", "i2f_weights", "i2o_weights",
                                  "r2i_weights", "r2c_weights", "r2f_weights", "r2o_weights",
                                  "c2i_weights", "c2f_weights", "c2o_weights",
                                  "igate_bias", "cgate_bias", "fgate_bias", "ogate_bias",
                                  "projection_weight", "projection_bias",
                                  "iactivationstateTensor", "icellstatetensor",
                                  "ilayer_norm_coefficients",
                                  "flayer_norm_coefficients",
                                  "clayer_norm_coefficients",
                                  "olayer_norm_coefficients"};

const int inputdims[24] = {3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1};

int inputIfenable[24] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

int count_num()
{
    int count = 0;
    for(int i = 0; i < 24; i++)
    {
        count += inputIfenable[i];
    }

    return count;
}

int create_1dim_inputnode(graph_t graph, const char* node_name, int* dim, int data_type, const char* Op_name)
{
    //const char* Opname = "InputOp";
    //char finalOpname[100];
    //strcat(finalOpname, Opname);
    node_t node = create_graph_node(graph, node_name, Op_name);
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);
    if(tensor == nullptr)
    {
        release_graph_node(node);
        std::cout<<"create input node"<<node_name<<" fail."<<std::endl;

        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    set_tensor_shape(tensor, dim, 1);
    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_2dim_inputnode(graph_t graph, const char* node_name, int* dims, int data_type, const char* Op_name)
{
    //const char* Opname = "InputOp\0";
    //char finalOpname[100];
    //strcat(finalOpname, Opname);
    node_t node = create_graph_node(graph, node_name, Op_name);
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);
    if(tensor == nullptr)
    {
        release_graph_node(node);
    
        std::cout<<"create input node"<<node_name<<" fail."<<std::endl;

        return -1;
    }   
    
    if(node == nullptr)
    {
        std::cout<<"create node fail"<<std::endl;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    set_tensor_shape(tensor, dims, 2);
    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_3dim_inputnode(graph_t graph, const char* node_name, int* dims, int data_type, const char* Op_name)
{
    node_t node = create_graph_node(graph, node_name, Op_name);
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        std::cout<<"create input node fail."<<std::endl;

        return -1;
    }   

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    set_tensor_shape(tensor, dims, 3);
    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

void create_inputname_vect(const char** inputnamesvec)
{
    int count = 0;
    for(int i = 0; i < 24; i++)
    {
        if(inputIfenable[i])
        {
            inputnamesvec[count] = inputTensorNames[i];
            count += 1;
        }
    }
    return ;
}

int create_test_node(graph_t graph, const char* node_name, int batch_size, int output_size, int cell_size, int sequence_size)
{

    node_t test_node = create_graph_node(graph, node_name, "LayerNormLSTM");
    
    const float proj_clip = 0.0;
    const float cell_clip = 0.0;
    const TEngine::FusedActivation activation_function = TEngine::FusedActivation::kTanh;
    const TEngine::KernelType kerneltype = TEngine::KernelType::kFullKernel;
      
    set_node_attr_int(test_node, "output_size", &output_size);
    set_node_attr_int(test_node, "hidden_size", &cell_size);
    set_node_attr_float(test_node, "proj_clip", &proj_clip);
    set_node_attr_float(test_node, "cell_clip", &cell_clip);
    set_node_attr_pointer(test_node, "kernel_type", &kerneltype);
    set_node_attr_pointer(test_node, "fused_activation", &activation_function);

    int output_data_type = 0;
    int count = 0;
    for(int i = 0; i < 24; i++)
    {
        if(inputIfenable[i])
        {
            tensor_t input_tensor = get_graph_tensor(graph, inputTensorNames[i]);
            int data_type = get_tensor_data_type(input_tensor);

            if(input_tensor == nullptr)
            {
                std::cout << "ERRNO: " << get_tengine_errno() << "\n";
                return -1;
            }
            //set the output data type to input data type
            if(0 == i)
            {
                output_data_type = data_type; 
            }

            set_node_input_tensor(test_node, count, input_tensor);
            //release_graph_node(input_tensor);
            count++;
        }
    }

    tensor_t output_tensor = create_graph_tensor(graph, node_name, output_data_type);
    
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);
    
    release_graph_tensor(output_tensor);

    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph(const char* test_node_name, int batch_size, int input_size, 
                          int cell_size, int output_size, int sequence_size, int layout, int data_type)
{
    const std::vector<std::vector<int>> input_sizes = {
          {batch_size, sequence_size, input_size},  // input tensor

          {cell_size, input_size},  // input_to_input_weight tensor
          {cell_size, input_size},  // input_to_forget_weight tensor
          {cell_size, input_size},  // input_to_cell_weight tensor
          {cell_size, input_size},  // input_to_output_weight tensor

          {cell_size, output_size},  // recurrent_to_input_weight tensor
          {cell_size, output_size},  // recurrent_to_forget_weight tensor
          {cell_size, output_size},  // recurrent_to_cell_weight tensor
          {cell_size, output_size},  // recurrent_to_output_weight tensor

          {cell_size},  // cell_to_input_weight tensor
          {cell_size},  // cell_to_forget_weight tensor
          {cell_size},  // cell_to_output_weight tensor

          {cell_size},  // input_gate_bias tensor
          {cell_size},  // forget_gate_bias tensor
          {cell_size},  // cell_bias tensor
          {cell_size},  // output_gate_bias tensor

          {output_size, cell_size},  // projection_weight tensor
          {0},                 // projection_bias tensor

          {batch_size, output_size},  // activation_state tensor
          {batch_size, cell_size},    // cell_state tensor

          {cell_size},  // input_layer_norm_coefficient tensor
          {cell_size},  // forget_layer_norm_coefficient tensor
          {cell_size},  // cell_layer_norm_coefficient tensor
          {cell_size},  // output_layer_norm_coefficient tensor
    };

    graph_t graph = create_graph(nullptr, nullptr, nullptr);

    if(graph == nullptr)
    {
        std::cerr << "create failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_layout(graph, layout) < 0)
    {
        std::cerr << "set layout failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* InputOpname = "InputOp";
    const char* ConstOpname = "Const";
    for(int i = 0; i < 24; i++)
    {
        int errnum = 0;
        int* dims = const_cast<int*>(input_sizes[i].data());
        //std::cout<<i<<std::endl;
        switch (inputdims[i])
        {
            
            case 1:
                if(0 != *dims)
                {
                    errnum = create_1dim_inputnode(graph, inputTensorNames[i],dims, data_type, ConstOpname);
                }
                else
                {
                   inputIfenable[i] = 0;
                }
                //std::cout<<i<<std::endl;
                break;

            case 2:
                if(0 != *dims)
                {
                    errnum = create_2dim_inputnode(graph, inputTensorNames[i],dims, data_type, ConstOpname);                
                }
                else
                {
                    inputIfenable[i] = 0;
                }
                //std::cout<<i<<std::endl;
                break;
            
            case 3:
                if(0 != *dims)
                {
                    errnum = create_3dim_inputnode(graph, inputTensorNames[i],dims, data_type, InputOpname);
                }
                else
                {
                    inputIfenable[i] = 0;
                }
                break;

            default:
                break;
        }

        if(errnum < 0)
        {
            std::cerr << "create input failed\n";
            return nullptr;
        }
    }

    if(create_test_node(graph, test_node_name, batch_size, output_size, cell_size, sequence_size) < 0)
    {
        std::cerr << "create test node failed\n";
        return nullptr;
    }
    int count = 0;

    for(int i = 0; i < 24; i++)
    {
        count += inputIfenable[i];
    }

    const char* inputs[23] = {};
    const char* outputs[] = {test_node_name};
    
    create_inputname_vect(inputs);
    
    //for(int k = 0; k < count; k++)
    //{
    //    std::cout<<inputs[k]<<std::endl;
    //}

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

float** set_input_data(graph_t graph, const std::vector<std::vector<float>>& input)
{
    int count = 0;
    int enable_count = 0;
    
    for(int i = 0; i < 24; i++)
    {
        enable_count += inputIfenable[i];
    }
    float** i_bufs = (float **) malloc((enable_count+1)*sizeof(float*));

    for(int i = 0; i < 24; i++)
    {
        if(inputIfenable[i])
        {
            
            tensor_t input_tensor = get_graph_input_tensor(graph, count, 0);

            int buf_size = get_tensor_buffer_size(input_tensor);

            float* i_buf = (float*)malloc(buf_size);

            i_bufs[i] = i_buf;

            // int *dims = (int*)malloc(3);

            int dims[inputdims[i]];

            get_tensor_shape(input_tensor, dims, inputdims[i]);

            int elem_num = 1;
            for(int n = 0; n < inputdims[i]; n++)
            {
                elem_num *= dims[n];
            }

            float* input_data = const_cast<float*>(input[i].data());
            
            for (int n = 0; n < elem_num; n++)
            {
                // float* f = (float*) i_buf;
                i_buf[n] = input_data[n];    
            }
            set_tensor_buffer(input_tensor,i_buf, buf_size);
            release_graph_tensor(input_tensor);
            // free(dims);
            
            
            // free(i_buf);
            count++;
        }    
    }
    // free(i_bufs);
    return i_bufs;
}
void inline dump_kernel_value(node_t test_node, const char* dump_file_name)
{
    tensor_t tensor = get_node_output_tensor(test_node, 0);

    std::ofstream of(dump_file_name, std::ios::out);
    int kernel_dim[4];
    int dim_len = 0;
    dim_len = get_tensor_shape(tensor, kernel_dim, 4);
    int data_couts = 1;
    for(int ii = 0; ii < dim_len; ++ii)
    {
        data_couts *= kernel_dim[ii];
    }

    const float* tmp_data = ( const float* )get_tensor_buffer(tensor);
    char tmpBuf[1024];
    int iPos = 0;
    for(int ii = 0; ii < data_couts; ++ii)
    {
        iPos += sprintf(&tmpBuf[iPos], "%.18e", tmp_data[ii]);
        of << tmpBuf << std::endl;
        iPos = 0;
    }

    of.close();
    release_graph_tensor(tensor);
}
void dump_output_data(node_t test_node)
{
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);

    int dims[3];

    get_tensor_shape(output_tensor, dims, 3);

    void* o_buf = get_tensor_buffer(output_tensor);

    int elem_num = dims[0] * dims[1] * dims[2];
    
    for(int i = 0; i < elem_num; i++)
    {
        float* p = ( float* )o_buf;
        std::cout << i << " " << p[i] << "\n";
    }
    release_graph_tensor(output_tensor);
    // free(dims);
}
void getdatacompare(const char* filename,const char* filename1)
{
    char buffer[256];
    char buffer1[256];
    std::fstream outfile;
    std::fstream outfile1;
    std::vector<float> f_vec={};
    std::vector<float> f_vec1={};
    outfile.open(filename,std::ios::in);

    while(outfile.getline(buffer,256))
    {
        f_vec.push_back(atof(buffer));
    }
    outfile1.open(filename1,std::ios::in);
    while(outfile1.getline(buffer1,256))
    {
        f_vec1.push_back(atof(buffer1));
    }
    float losssum=0;
    for(unsigned int i=0;i<f_vec.size();i++)
    {
        losssum+=fabs((f_vec[i]-f_vec1[i]));
    }
    float avg_loos_rate=losssum/f_vec.size();
    if(avg_loos_rate<=1.e-04)
    {
        std::cout<<"pass\n";
    }else
    {
        std::cout<<"fail\n";
    }
    outfile.close();
    outfile1.close();
}
int main()
{
    const char* test_node_name = "test";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NHWC;
    int sequence_size = 3;
    int batch_size = 2;
    int cell_size = 4;
    int input_size = 5;
    int output_size = 3;

    init_tengine();

    graph_t graph = create_test_graph(test_node_name, batch_size, input_size, cell_size, output_size, sequence_size, layout, data_type);
    

    if(graph == nullptr)
        return 1;
    std::vector<std::vector<float>> param_input_data = {
        //input
         {// 2(Batch) * 3 (input_sequence_size) * 5 (n_input)
         0.7, 0.8, 0.1, 0.2, 0.3,   // seq 0
         0.8, 0.1, 0.2, 0.4, 0.5,   // seq 1
         0.2, 0.7, 0.7, 0.1, 0.7,   // seq 2
         0.3, 0.2, 0.9, 0.8, 0.1,   // seq 0
         0.1, 0.5, 0.2, 0.4, 0.2,   // seq 1
         0.6, 0.9, 0.2, 0.5, 0.7},  // seq 2
        
        //input_to_input_weights
        {
        0.5,  0.6,  0.7,  -0.8, -0.9, 0.1,  0.2,
        0.3,  -0.4, 0.5,  -0.8, 0.7,  -0.6, 0.5,
        -0.4, -0.5, -0.4, -0.3, -0.2, -0.1},

        //input_to_cell_weights
        {
        -0.4, -0.3, -0.2, -0.1, -0.5, 0.5,  -0.2,
        -0.3, -0.2, -0.6, 0.6,  -0.1, -0.4, -0.3,
        -0.7, 0.7,  -0.9, -0.5, 0.8,  0.6},

        //input_to_forget_weights
        {
        -0.6, -0.1, 0.3,  0.2,  0.9,  -0.5, -0.2,
        -0.4, 0.3,  -0.8, -0.4, 0.3,  -0.5, -0.4,
        -0.6, 0.3,  -0.4, -0.6, -0.5, -0.5},

        //input_to_output_weights
        {
        -0.8, -0.4, -0.2, -0.9, -0.1, -0.7, 0.3,
        -0.3, -0.8, -0.2, 0.6,  -0.2, 0.4,  -0.7,
        -0.3, -0.5, 0.1,  0.5,  -0.6, -0.4},

        //recurrent_to_input_weights
        {
        -0.2, -0.3, 0.4,  0.1,  -0.5, 0.9,
        -0.2, -0.3, -0.7, 0.05, -0.2, -0.6},

        //recurrent_to_cell_weights
        {
        -0.3, 0.2, 0.1, -0.3, 0.8,  -0.08,
        -0.2, 0.3, 0.8, -0.6, -0.1, 0.2},

        //recurrent_to_forget_weights
        {
        -0.5, -0.3, -0.5, -0.2, 0.6, 0.4,
        0.9,  0.3,  -0.1, 0.2,  0.5, 0.2},

        //recurrent_to_output_weights
        {
        0.3,  -0.1, 0.1,  -0.2, -0.5, -0.7,
        -0.2, -0.6, -0.1, -0.4, -0.7, -0.2},

        //cell_to_input_weights
        {0.05, 0.1, 0.25, 0.15},

        //cell_to_forget_weights
        {-0.02, -0.15, -0.25, -0.03},

        //cell_to_output_weights
        {0.1, -0.1, -0.5, 0.05},

        //input_gate_bias
        {0.03, 0.15, 0.22, 0.38},

        //cell_gate_bias
        {-0.05, 0.72, 0.25, 0.08},

        //forget_gate_bias
        {0.1, -0.3, -0.2, 0.1},

        //output_gate_bias
        {0.05, -0.01, 0.2, 0.1},

        //projection_weight
        {
        -0.1, 0.2,  0.01, -0.2, 0.1,  0.5,
        0.3,  0.08, 0.07, 0.2,  -0.4, 0.2},

        //projection_bias
        {},

        //activationstate
        {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0},
        
        //icellstate
        {
        0.0, 0.0, 0.0, 0.0,   
        0.0, 0.0, 0.0, 0.0},

        //input_layer_norm_coefficients
        {0.1, 0.2, 0.3, 0.5},
        //forget_layer_norm_coefficients
        {0.2, 0.2, 0.4, 0.3},
        //cell_layer_norm_coefficients
        {0.7, 0.2, 0.3, 0.8},
        //output_layer_norm_coefficients
        {0.6, 0.2, 0.2, 0.5}
    };
    
    float** i_bufs=set_input_data(graph, param_input_data);
    if(prerun_graph(graph) < 0)
    {   
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    node_t test_node = get_graph_node(graph, test_node_name);



    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }
    //dump_graph(graph);
    dump_kernel_value(test_node,"./out/test_tflite_lnlstm.txt"); 
    getdatacompare("./out/test_tflite_lnlstm.txt","./data/lnlstm_tflite_out.txt");  
    // dump_output_data(test_node);
    free(i_bufs);
    // release_graph_node(test_node);
    postrun_graph(graph);
    
    destroy_graph(graph);
    release_tengine();
    
    //for(int i = 0; i < count_num(); i++)
    //{
    //   free(i_bufs[i]);
    //}
    
    return 0;
}
