   #include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"
#include "common_util.hpp"
#include "image_process.hpp"
#include <sys/time.h>
//const char* text_file = "./models/1.json";
const char* image_file = "./tests/images/cat.jpg";
const char* label_file = "./models/synset_words.txt";

const float channel_mean[3] = { 0.485, 0.456, 0.406};
const float scale[3] = {0.229,0.224,0.225};
using namespace TEngine;

void get_input_data_mx(const char* image_file, float* input_data, int img_h, int img_w, int img_c, const float* mean, const float* scale)
{

    float scales[3] = {scale[0], scale[1], scale[2]};   
    float means[3] = {mean[0], mean[1], mean[2]};
    image img = imread(image_file, img_w, img_h, means, scales, MXNET);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h); 
    
}




int repeat_count = 100;

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}


int main(int argc, char* argv[])
{
    std::string text_file = "./models/mx_mobile_post.tmfile";
    int res; 
    while((res = getopt(argc, argv, "r:m:")) != -1)
    {
        switch(res)
        {
            case 'r':
                repeat_count = strtoul(optarg, NULL, 10);
                break;
            case 'm':
                text_file  = optarg;
                break;
            default:
                break;
        }
    }

    // const char * model_name="mobilenet";
    int img_h = 224;
    int img_w = 224;

    init_tengine();

    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(nullptr, "tengine", text_file.c_str());
    //dump_graph_tensor_scale(graph);
    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return -1;
    }

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;

    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);

    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }

    int dims[] = {1, 3, img_h, img_w};

    set_tensor_shape(input_tensor, dims, 4);

    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3); 
    //get_input_data(image_file, input_data, img_h, img_w, channel_mean, 0.017);
    get_input_data_mx(image_file, input_data, img_h, img_w,3, channel_mean, scale);
    int img_size = img_h * img_w * 3;
    float in_scale = 0;
    int in_zero = 0;
    get_tensor_quant_param(input_tensor,&in_scale,&in_zero,1);
    printf("intput scale is %f,input zero point is %d\n",in_scale,in_zero);
    //quant the input data
    int8_t * input_s8 = (int8_t*)malloc(sizeof(int8_t) * img_h * img_w * 3); 
    for(int i = 0; i < img_size; ++i)
    {   
        input_s8[i] = round(input_data[i] / in_scale);
    }   
    //set the input data type 
    set_tensor_data_type(input_tensor,TENGINE_DT_INT8);

    /* setup input buffer */

    if(set_tensor_buffer(input_tensor, input_s8, 3 * img_h * img_w) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    /* run the graph */
    int ret_prerun = prerun_graph(graph);
    
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }
    // benchmark start here
    printf("REPEAT COUNT= %d\n", repeat_count);

    unsigned long start_time = get_cur_time();

    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);

    unsigned long end_time = get_cur_time();

    unsigned long off_time = end_time - start_time;
    std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count,
                off_time);

    tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);
    if(NULL == output_tensor)
    {
        std::cout << "output tensor is NULL\n";
    }

    int8_t* out_data_s8 = (int8_t* )(get_tensor_buffer(output_tensor));
    int count = get_tensor_buffer_size(output_tensor);
    float * out_data_fp32 = (float*) malloc(count * sizeof(float));
    float out_scale = 1.f;
    int out_zero = 0;

    get_tensor_quant_param(output_tensor,&out_scale,&out_zero,1);
    printf("out scale is %f\n",out_scale);

    //dequant the output data
    for(int i = 0; i < count ; i ++)
    {
        out_data_fp32[i] = out_data_s8[i] * out_scale;
    }
    float* end = out_data_fp32 + count;

    std::vector<float> result(out_data_fp32, end);

    std::vector<int> top_N = Argmax(result, 5);

    std::vector<std::string> labels;

    LoadLabelFile(labels, label_file);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
        std::cout << labels[idx] << "\"\n";
    }

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    postrun_graph(graph);

    destroy_graph(graph);

    free(input_data);
    free(input_s8);
    free(out_data_fp32);


    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
