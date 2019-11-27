/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */

#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "cpu_device.h"

#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

std::atomic<int> thread_done;
int thread_num = 0;
std::string image_file;
std::string cpu_2A72_save_name = "cpu_2A72";
std::string cpu_4A53_save_name = "cpu_4A53";
std::string gpu_save_name = "gpu";

int cpu_2A72_repeat_count = 120;
int gpu_repeat_count = 105;
int cpu_4A53_repeat_count = 95;

volatile int barrier = 1;

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};

void get_input_data_ssd(std::string& image_file, float* input_data, int img_h, int img_w)
{
    image img = imread(image_file.c_str());

    image resImg = resize_image(img, img_w, img_h);
    resImg = rgb2bgr_premute(resImg);
    float* img_data = ( float* )resImg.data;
    int hw = img_h * img_w;

    float mean[3] = {127.5, 127.5, 127.5};
    for(int c = 0; c < 3; c++)
    {
        for(int h = 0; h < img_h; h++)
        {
            for(int w = 0; w < img_w; w++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843 * (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}

void post_process_ssd(std::string& image_file, float threshold, float* outdata, int num, const std::string& save_name)
{
    const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                 "sofa",       "train",     "tvmonitor"};

    image im = imread(image_file.c_str());

    int raw_h = im.h;
    int raw_w = im.w;
    std::vector<Box> boxes;
    printf("detect result num: %d \n", num);
    for(int i = 0; i < num; i++)
    {
        if(outdata[1] >= threshold)
        {
            Box box;
            box.class_idx = outdata[0];
            box.score = outdata[1];
            box.x0 = outdata[2] * raw_w;
            box.y0 = outdata[3] * raw_h;
            box.x1 = outdata[4] * raw_w;
            box.y1 = outdata[5] * raw_h;
            boxes.push_back(box);
            printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
            printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
        }
        outdata += 6;
    }
    for(int i = 0; i < ( int )boxes.size(); i++)
    {
        Box box = boxes[i];

        std::ostringstream score_str;
        score_str << box.score * 100;
        std::string labelstr = std::string(class_names[box.class_idx]) + " : " + score_str.str();

        put_label(im, labelstr.c_str(), 0.02, box.x0, box.y0, 255, 255, 125);
        draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 125, 0, 125);
    }

    save_image(im, "Mobilenet_SSD.jpg");

    std::cout << "======================================\n";
    std::cout << "[DETECTED IMAGE SAVED]:\t"
              << "Mobilenet_SSD"
              << "\n";
    std::cout << "======================================\n";
}

void run_test(graph_t graph, const std::string& save_name, int repeat_count, float* avg_time)
{
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);

    if(input_tensor == nullptr)
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
        return;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    int ret_prerun = prerun_graph(graph);
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return;
    }

    if(save_name == "gpu")
    {
        // warm up
        get_input_data_ssd(image_file, input_data, img_h, img_w);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
        run_graph(graph, 1);
        barrier = 0;
    }
    else
    {
        while(barrier)
            ;
    }

    struct timeval t0, t1;
    float total_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        get_input_data_ssd(image_file, input_data, img_h, img_w);

        gettimeofday(&t0, NULL);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
        run_graph(graph, 1);

        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
    }

    std::cout << "--------------------------------------\n";
    std::cout << save_name << ": repeat " << repeat_count << " times, avg " << total_time / repeat_count
              << " ms  all: " << total_time << "ms\n";
    (*avg_time) = total_time / repeat_count;
    tensor_t out_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out");
    int out_dim[4];
    get_tensor_shape(out_tensor, out_dim, 4);
    float* outdata = ( float* )get_tensor_buffer(out_tensor);
    int num = out_dim[1];
    float show_threshold = 0.5;

    post_process_ssd(image_file, show_threshold, outdata, num, save_name + "_save.jpg");

    release_graph_tensor(out_tensor);
    release_graph_tensor(input_tensor);

    postrun_graph(graph);
    free(input_data);
    destroy_graph(graph);
}

void cpu_thread_a53(const char* pproto_file, const char* pmodel_file, float* avg_time)
{
    graph_t graph = create_graph(NULL, "caffe", pproto_file, pmodel_file);
    if(graph == nullptr)
    {
        thread_done++;
        return;
    }

    if(set_graph_device(graph, "a53") < 0)
    {
        std::cerr << "set device a53 failed\n";
    }

    run_test(graph, cpu_4A53_save_name, cpu_4A53_repeat_count, avg_time);
    thread_done++;
}
void cpu_thread_a72(const char* pproto_file, const char* pmodel_file, float* avg_time)
{
    graph_t graph = create_graph(NULL, "caffe", pproto_file, pmodel_file);
    if(graph == nullptr)
    {
        thread_done++;
        return;
    }

    if(set_graph_device(graph, "a72") < 0)
    {
        std::cerr << "set device a72 failed\n";
    }

    run_test(graph, cpu_2A72_save_name, cpu_2A72_repeat_count, avg_time);
    thread_done++;
}

void gpu_thread(const char* pproto_file, const char* pmodel_file, float* avg_time)
{
    graph_t graph = create_graph(NULL, "caffe", pproto_file, pmodel_file);

    if(graph == nullptr)
    {
        thread_done++;
        return;
    }

    set_graph_device(graph, "acl_opencl");

    run_test(graph, gpu_save_name, gpu_repeat_count, avg_time);
    thread_done++;
}

int main(int argc, char* argv[])
{
    const std::string root_path;
    std::string proto_file;
    std::string model_file;
    const char* pproto_file;
    const char* pmodel_file;

    int res;
    while((res = getopt(argc, argv, "p:m:i:hd:")) != -1)
    {
        switch(res)
        {
            case 'p':
                proto_file = optarg;
                break;
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'h':
                std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                          << "   [-p proto_file] [-m model_file] [-i image_file]\n";
                return 0;
            default:
                break;
        }
    }

    if(proto_file.empty())
    {
        proto_file = root_path + DEF_PROTO;
        std::cout << "proto file not specified,using " << proto_file << " by default\n";
    }
    if(model_file.empty())
    {
        model_file = root_path + DEF_MODEL;
        std::cout << "model file not specified,using " << model_file << " by default\n";
    }
    if(image_file.empty())
    {
        image_file = root_path + DEF_IMAGE;
        std::cout << "image file not specified,using " << image_file << " by default\n";
    }

    /* do not let GPU run concat */
    setenv("GPU_CONCAT", "0", 1);
    /* using GPU fp16 */
    setenv("ACL_FP16", "1", 1);
    /* default CPU device using 0,1,2,3 */
    setenv("TENGINE_CPU_LIST", "2", 1);
    /* using fp32 or int8 */
    setenv("KERNEL_MODE", "2", 1);

    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return -1;
    // collect avg_time for each case
    float avg_times[3] = {0., 0., 0.};

    // thread 0 for cpu 2A72
    const struct cpu_info* p_info = get_predefined_cpu("rk3399");
    int a72_list[] = {4, 5};
    set_online_cpu(( struct cpu_info* )p_info, a72_list, sizeof(a72_list) / sizeof(int));
    create_cpu_device("a72", p_info);

    // thread 3 for cpu 4A53
    const struct cpu_info* p_info1 = get_predefined_cpu("rk3399");
    int a53_list[] = {0, 1, 2, 3};
    set_online_cpu(( struct cpu_info* )p_info1, a53_list, sizeof(a53_list) / sizeof(int));
    create_cpu_device("a53", p_info1);
#if 0
    if (load_model(model_name, "caffe", proto_file.c_str(), model_file.c_str()) < 0)
    {
        std::cout<<"load model failed\n";
        return 1;
    }
    std::cout << "load model done!\n";
#endif
    pproto_file = proto_file.c_str();
    pmodel_file = model_file.c_str();
    thread_done = 0;
    std::thread* t0 = new std::thread(cpu_thread_a72, pproto_file, pmodel_file, &avg_times[0]);
    thread_num++;

    // thread 1 for gpu +1 A53
    std::thread* t1 = new std::thread(gpu_thread, pproto_file, pmodel_file, &avg_times[1]);
    thread_num++;

    std::thread* t2 = new std::thread(cpu_thread_a53, pproto_file, pmodel_file, &avg_times[2]);
    thread_num++;

    t0->join();
    delete t0;

    t1->join();
    delete t1;

    t2->join();
    delete t2;

    std::cout << "thread_done: " << ( int )thread_done << "\ntest done\n";

    std::cout << "=================================================\n";
    std::cout << " Using 3 thread, MSSD performance "
              << (1000. / avg_times[0] + 1000. / avg_times[1] + 1000. / avg_times[2]) << "  FPS \n";
    std::cout << "=================================================\n";

    release_tengine();

    return 0;
}
