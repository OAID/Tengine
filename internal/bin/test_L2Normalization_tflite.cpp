#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include <math.h>

#include "tengine_c_api.h"

const char* model_path = "./models/l2norm.tflite";
const std::vector<float> test_data1 = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  
                                       -1.1, 0.6, 0.7, 1.2, -0.7, 0.1};

const std::vector<float> test_data2 = {-1.9, 0.5, 1.4, 0.3,
                                       -1.9, 0.5, 1.4, 0.3,
                                       -1.9, 0.5, 1.4, 0.3};

const std::vector<float> test_data3 = {-1.2, 0.9, 1.5,
                                       -1.2, 0.9, 1.5,
                                       -1.2, 0.9, 1.5,
                                       -1.2, 0.9, 1.5};

void set_input_data(float* input_data, int num, int size)
{
    switch (num)
    {
    case 1:
        for(int i = 0; i < size; i++)
            *input_data++ = test_data1[i];
        break;
    case 2:
        for(int i = 0; i < size; i++)
            *input_data++ = test_data2[i];
        break;
    case 3:
        for(int i = 0; i < size; i++)
            *input_data++ = test_data3[i];
        break;
    default:
        break;
    }
}
void inline dump_kernel_value(const tensor_t tensor, const char* dump_file_name)
{
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

    graph_t graph = create_graph(0, "tflite", model_path);
    if(graph == nullptr)
    {
        std::cout << "create graph failed!"<<std::endl;
        return 1;
    }
    std::cout << "create graph done!"<<std::endl;

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(!check_tensor_valid(input_tensor))
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
        return 1;
    }

    int h = 2;
    int w = 2;
    int flatten_size = h * w * 3;

    int dims[] = {1, h, w, 3};
    set_tensor_shape(input_tensor, dims, 4);
    if(prerun_graph(graph) != 0)
    {
        std::cout << "prerun _graph failed\n";
        return -1;
    }
    float* input_data = (float*)malloc(sizeof(float)*flatten_size);
   
    set_input_data(input_data, 3, flatten_size);
    set_tensor_buffer(input_tensor, input_data, flatten_size);
    if(run_graph(graph, 1) != 0)
    {
        std::cout << "run _graph failed\n";
        return -1;
    }

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    dump_kernel_value(output_tensor,"./out/test_tflite_l2norm.txt");

    getdatacompare("./out/test_tflite_l2norm.txt","./data/l2norm_tflite_out.txt");


    if(postrun_graph(graph) != 0)
    {
        std::cout << "postrun graph failed\n";
    }
    free(input_data);

    destroy_graph(graph);

    return 0;
    
}

