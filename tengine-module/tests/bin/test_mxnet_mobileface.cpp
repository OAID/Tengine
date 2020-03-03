#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "common_util.hpp"

const char* mxnet_text_file = "./models/model-symbol.json";
const char* mxnet_model_file = "./models/model-mobilefacenet-128-0077.params";

const char* caffe_text_file = "./models/model-77.prototxt";
const char* caffe_model_file = "./models/model-77.caffemodel";

const char* image_file = "./tests/images/mobileface01.jpg";
// const char* image_file = "./tests/images/mobileface02.jpg";

using namespace TEngine;

int repeat_count = 1;
int img_h = 112;
int img_w = 112;

const float mxnet_mean[3] = {0.0, 0.0, 0.0};
const float mxnet_scale = 1.0;

const float caffe_mean[3] = {127.5, 127.5, 127.5};
const float caffe_scale = 0.00781;

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
    image img = imread(image_file);

    image resImg = resize_image(img, img_w, img_h);
    resImg = rgb2bgr_premute(resImg);
    float* img_data = ( float* )resImg.data;

    int hw = img_h * img_w;
    for(int c = 0; c < 3; c++)
        for(int h = 0; h < img_h; h++)
            for(int w = 0; w < img_w; w++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }
}

void dump_float(const char* fname, float* data, int number)
{
    FILE* fp = fopen(fname, "w");

    for(int i = 0; i < number; i++)
        fprintf(fp, " %f\n", data[i]);

    fclose(fp);
}

int test_graph(graph_t graph, const char* dump_file, const float means[], const float scale)
{
    /* get input tensor */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor of graph\n");
        return -1;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);

    /* setup input buffer */
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);
    get_input_data(image_file, input_data, img_h, img_w, means, scale);

    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
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
    // dump_graph(graph);

    unsigned long start_time = get_cur_time();

    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);

    unsigned long end_time = get_cur_time();

    unsigned long off_time = end_time - start_time;
    std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count,
                off_time);

    /* get output tensor */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    if(output_tensor == nullptr)
    {
        std::printf("Cannot find output tensor\n");
        return -1;
    }

    int size = get_tensor_buffer_size(output_tensor);
    std::printf("output tensor buffer size: %d\n", size);

    float* data1 = ( float* )get_tensor_buffer(output_tensor);

    dump_float(dump_file, data1, size / 4);

    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    return 0;
}

int main(int argc, char* argv[])
{
    int res;
    while((res = getopt(argc, argv, "r:")) != -1)
    {
        switch(res)
        {
            case 'r':
                repeat_count = strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }

    init_tengine();

    std::cout << "Tengine version: " << get_tengine_version() << "\n";

    if(request_tengine_version("1.0") < 0)
        return 1;

    graph_t mxnet_graph = create_graph(nullptr, "mxnet", mxnet_text_file, mxnet_model_file);
    if(mxnet_graph == nullptr)
    {
        std::cout << "Create mxnet_graph failed\n";
        return -1;
    }

    graph_t caffe_graph = create_graph(nullptr, "caffe", caffe_text_file, caffe_model_file);
    if(caffe_graph == nullptr)
    {
        std::cout << "Create caffe_graph failed\n";
        return -1;
    }

    std::cout << "Test mxnet graph:\n";
    if(test_graph(mxnet_graph, "./mxnet_output_data.txt", mxnet_mean, mxnet_scale) < 0)
        return -1;

    std::cout << "Test caffe graph:\n";
    if(test_graph(caffe_graph, "./caffe_output_data.txt", caffe_mean, caffe_scale) < 0)
        return -1;

    postrun_graph(mxnet_graph);
    postrun_graph(caffe_graph);
    destroy_graph(mxnet_graph);
    destroy_graph(caffe_graph);
    release_tengine();

    std::cout << "ALL TEST DONE\n";
    return 0;
}
