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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_MAX_BOX_COUNT 100
#define DEFAULT_REPEAT_COUNT    1
#define DEFAULT_THREAD_COUNT    1

typedef struct Box
{
    int x0;
    int y0;
    int x1;
    int y1;
    int class_idx;
    float score;
} Box_t;

void post_process_ssd(const char* image_file, float threshold, const float* outdata, int num)
{
    const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                 "sofa",       "train",     "tvmonitor"};

    image im = imread(image_file);

    int raw_h = im.h;
    int raw_w = im.w;

    Box_t* boxes = malloc(sizeof(Box_t) * DEFAULT_MAX_BOX_COUNT);
    int box_count = 0;

    fprintf(stderr, "detect result num: %d \n", num);
    for (int i = 0; i < num; i++)
    {
        if (outdata[1] >= threshold)
        {
            Box_t box;

            box.class_idx = outdata[0];
            box.score = outdata[1];
            box.x0 = outdata[2] * raw_w;
            box.y0 = outdata[3] * raw_h;
            box.x1 = outdata[4] * raw_w;
            box.y1 = outdata[5] * raw_h;

            boxes = realloc(boxes, sizeof(Box_t) * (box_count + 1));
            boxes[box_count] = box;
            box_count++;

            fprintf(stderr, "%s\t:%.1f%%\n", class_names[box.class_idx], box.score * 100);
            fprintf(stderr, "BOX:( %d , %d ),( %d , %d )\n", box.x0, box.y0, box.x1, box.y1);
        }
        outdata += 6;
    }
    for (int i = 0; i < box_count; i++)
    {
        Box_t box = boxes[i];
        draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 125, 0, 125);
    }

    free(boxes);

    save_image(im, "tengine_example_out");
    free_image(im);
    fprintf(stderr, "======================================\n");
    fprintf(stderr, "[DETECTED IMAGE SAVED]:\n");
    fprintf(stderr, "======================================\n");
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = NULL;
    char* image_file = NULL;
    int img_h = 300;
    int img_w = 300;
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float scale[3] = {0.007843f, 0.007843f, 0.007843f};
    float show_threshold = 0.5f;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'r':
                repeat_count = atoi(optarg);
                break;
            case 't':
                num_thread = atoi(optarg);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (model_file == NULL)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == NULL)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (graph == NULL)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w};    // nchw
    float* input_data = ( float* )malloc(img_size * sizeof(float));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }    

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun graph failed\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_data(image_file, input_data, img_h, img_w, mean, scale);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out"
    int out_dim[4];
    get_tensor_shape(output_tensor, out_dim, 4);
    float* output_data = ( float* )get_tensor_buffer(output_tensor);
    post_process_ssd(image_file, show_threshold, output_data, out_dim[1]);

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
