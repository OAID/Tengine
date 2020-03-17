#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include "tengine_c_api.h"
#include "common.hpp"
#include "tengine_operations.h"

#define DEF_MODEL "models/mobilenet_ssd.tflite"
#define DEF_IMAGE "images/ssd_dog.jpg"
#define DEF_LABEL "models/coco_labels_list.txt"

void get_input_data_ssd(const char* image_file, float* input_data, int img_h, int img_w)
{
    float mean[3] = {127.5, 127.5, 127.5};
    float scales[3] = {0.007843, 0.007843, 0.007843};
    image img = imread(image_file, img_w, img_h, mean, scales, CAFFE);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h);   
    free_image(img); 
}

void post_process_ssd(const char* label_image, tensor_t concat0, tensor_t concat1)
{
    float* score_ptr = ( float* )get_tensor_buffer(concat1);

    int dims[4] = {0};
    int dims_size = get_tensor_shape(concat0, dims, 4);
    std::cout << "box shape: [";
    for(int i = 0; i < dims_size; i++)
        std::cout << dims[i] << ",";
    std::cout << "]\n";

    int num_boxes = dims[1];
    int num_classes = 90;

    std::vector<std::string> labels;
    LoadLabelFile(labels, label_image);

    for(int j = 0; j < num_boxes; j++)
    {
        float max_score = 0.f;
        int class_idx = 0;
        for(int i = 0; i < num_classes; i++)
        {
            float score = score_ptr[j * (num_classes + 1) + i + 1];
            if(score > max_score)
            {
                max_score = score;
                class_idx = i + 1;
            }
        }
        if(max_score > 0.6)
        {
            std::cout << "score: " << max_score << " class: " << labels[class_idx] << "\n";
        }
    }
}

int main(int argc, char* argv[])
{
    //const std::string root_path = get_root_path();
    //std::string model_file = root_path + DEF_MODEL;
    //std::string label_file = root_path + DEF_LABEL;
    //std::string image_file = root_path + DEF_IMAGE;
    
    int res;
    std::string model_file;
    std::string image_file;
    std::string label_file;

    while((res = getopt(argc, argv, "l:m:i:h")) != -1)
    {
	switch(res)
	{
		case 'm':
			model_file = optarg;
			break;
		case 'l':
			label_file = optarg;
			break;
		case 'i':
			image_file = optarg;
			break;
		case 'h':
			std::cout << "[Usage]: " << argv[0] << " [-h]\n"
				<< "   [-m model_file] [-i image_file] [-l label_file]\n";
			break;
		default:
			break;
	}
    }

    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(0, "tengine", model_file.c_str());
    if(graph == nullptr)
    {
        std::cout << "create graph failed!\n";
        return 1;
    }
    std::cout << "create graph done!\n";

    // input
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(!check_tensor_valid(input_tensor))
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
        return 1;
    }

    int dims[] = {1, img_h, img_w, 3};
    set_tensor_shape(input_tensor, dims, 4);
    if(prerun_graph(graph) != 0)
    {
        std::cout << "prerun _graph failed\n";
        return -1;
    }

    int repeat_count = 1;
    const char* repeat = std::getenv("REPEAT_COUNT");

    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    // warm up
    get_input_data_ssd(image_file.c_str(), input_data, img_h, img_w);
    set_tensor_buffer(input_tensor, input_data, img_size * 4);
    if(run_graph(graph, 1) != 0)
    {
        std::cout << "run _graph failed\n";
        return -1;
    }

        run_graph(graph, 1);
        run_graph(graph, 1);
        run_graph(graph, 1);
        run_graph(graph, 1);
        run_graph(graph, 1);
    struct timeval t0, t1;
    float total_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        run_graph(graph, 1);

        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
    }
    std::cout << "--------------------------------------\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n";

    tensor_t concat0 = get_graph_output_tensor(graph, 0, 0);
    tensor_t concat1 = get_graph_output_tensor(graph, 1, 0);

    post_process_ssd(label_file.c_str(), concat0, concat1);

    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(concat0);
    release_graph_tensor(concat1);

    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();

    return 0;
}

