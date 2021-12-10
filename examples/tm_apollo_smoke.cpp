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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hbshi@openailab.com
 */

#include <stdlib.h>
#include <stdio.h>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"
#include <vector>
#include <algorithm>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define DEFAULT_IMG_H        640
#define DEFAULT_IMG_W        960
#define DEFAULT_LOOP_COUNT   1
#define DEFAULT_THREAD_COUNT 4
#define DEFAULT_CPU_AFFINITY 0
#define MAX_DETECTION        50
#define PI                   3.14159265
#define NEG_PI               -3.14159265

// l h w
float pre_know_object_mean_dims[][3] = {{3.88000011, 1.63000000, 1.52999997},
                                        {1.77999997, 1.70000005, 0.57999998},
                                        {0.88000000, 1.73000002, 0.67000002}};

float camera_k_waymo[][3] = {{2.05556e+03, 0.00000e+00, 9.39658e+02},
                             {0.00000e+00, 2.05556e+03, 6.41072e+02},
                             {0.00000e+00, 0.00000e+00, 1.00000e+00}};

float camera_k_inv_waymo[][3] = {{0.00048649, 0., -0.45712993},
                                 {0., 0.00048649, -0.31187218},
                                 {0., 0., 1.}};

float box_3d_corner_map[8][3] = {
    {-0.5, -1, -0.5},
    {0.5, -1, -0.5},
    {0.5, 0, -0.5},
    {0.5, 0, 0.5},
    {0.5, -1, 0.5},
    {-0.5, -1, 0.5},
    {-0.5, 0, 0.5},
    {-0.5, 0, -0.5}};

int face_idx[][4] = {{5, 4, 3, 6}, {1, 2, 3, 4}, {1, 0, 7, 2}, {0, 5, 6, 7}};

struct box_3d_object
{
    float coo[8][2];
    int clas;
};

struct hm_process_object
{
    int pos;
    float score;
    int clas;
    float xs, ys;
};

struct reg_process_object
{
    float val[10];
};

struct post_process_object
{
    float score;
    int clas;
    float depth;
    float x, y, z;
    float dim0, dim1, dim2;
    float alpha_x, yaw;
    float x0, y0, x1, y1;
};

void process_hm_message(std::vector<hm_process_object>& hm_process_objects,
                        int c,
                        int h,
                        int w,
                        const float* hm_max_data,
                        const float* hm_data)
{
    for (int i = 0; i < c; ++i)
    {
        for (int j = 0; j < h * w; ++j)
        {
            if (hm_max_data[i * h * w + j] == hm_data[i * h * w + j])
            {
                hm_process_object object{};
                object.pos = j;
                object.score = hm_max_data[i * h * w + j];
                object.clas = i;
                object.xs = j % w;
                object.ys = j / w;
                hm_process_objects.push_back(object);
            }
        }
    }

    std::sort(hm_process_objects.begin(),
              hm_process_objects.end(),
              [](const hm_process_object& a, const hm_process_object& b) {
                  return a.score > b.score;
              });
}

void get_reg_data_object(const std::vector<hm_process_object>& hm_process_objects,
                         std::vector<reg_process_object>& reg_process_objects,
                         int h,
                         int w,
                         const float* reg_data)
{
    for (int i = 0; i < MAX_DETECTION; ++i)
    {
        reg_process_object object{};
        for (int j = 0; j < 10; ++j)
        {
            int index = j * h * w + hm_process_objects[i].pos;
            object.val[j] = reg_data[index];
        }
        reg_process_objects.push_back(object);
    }
}

void post_process(const std::vector<hm_process_object>& hm_process_objects,
                  const std::vector<reg_process_object>& reg_process_objects,
                  std::vector<post_process_object>& post_process_objects)
{
    for (int i = 0; i < MAX_DETECTION; ++i)
    {
        hm_process_object hm_object = hm_process_objects[i];
        if (hm_object.score < 0.25)
        {
            continue;
        }
        post_process_object object{};
        reg_process_object reg_object = reg_process_objects[i];
        object.score = hm_object.score;
        object.clas = hm_object.clas;
        object.depth = 16.31999 * reg_object.val[0] + 28.01;

        float tmp_x = (hm_object.xs + reg_object.val[1]) * 8;
        float tmp_y = (hm_object.ys + reg_object.val[2]) * 8;
        tmp_x *= object.depth;
        tmp_y *= object.depth;
        object.x = camera_k_inv_waymo[0][0] * tmp_x + camera_k_inv_waymo[0][2] * object.depth;
        object.y = camera_k_inv_waymo[1][1] * tmp_y + camera_k_inv_waymo[1][2] * object.depth;
        object.z = object.depth;

        int clas = hm_object.clas;
        // l h w
        float dim0 = pre_know_object_mean_dims[clas][0] * exp(reg_object.val[3]);
        float dim1 = pre_know_object_mean_dims[clas][1] * exp(reg_object.val[4]);
        float dim2 = pre_know_object_mean_dims[clas][2] * exp(reg_object.val[5]);
        object.y += dim1 / 2;
        object.dim0 = dim0;
        object.dim1 = dim1;
        object.dim2 = dim2;

        double ray = atan(object.x / (object.z + 1e-7));
        double alpha = atan(reg_object.val[6] / (reg_object.val[7] + 1e-7));
        if (reg_object.val[7] >= 0)
        {
            alpha = alpha - PI / 2;
        }
        else
        {
            alpha = alpha + PI / 2;
        }

        double yaw = alpha + ray;
        if (yaw > PI)
        {
            yaw -= 2 * PI;
        }
        else if (yaw < NEG_PI)
        {
            yaw += 2 * PI;
        }

        object.alpha_x = alpha;
        object.yaw = yaw;

        float x0 = hm_object.xs - reg_object.val[8] / 2;
        float y0 = hm_object.ys - reg_object.val[9] / 2;
        float x1 = hm_object.xs + reg_object.val[8] / 2;
        float y1 = hm_object.ys + reg_object.val[9] / 2;
        object.x0 = x0 * 8;
        object.y0 = y0 * 8;
        object.x1 = x1 * 8;
        object.y1 = y1 * 8;

        post_process_objects.push_back(object);
    }
}

void box_3d_process(const std::vector<post_process_object>& post_process_objects,
                    std::vector<box_3d_object>& box_3d_objects)
{
    for (int i = 0; i < post_process_objects.size(); ++i)
    {
        box_3d_object object{};
        // 8 points
        for (int j = 0; j < 8; ++j)
        {
            float tmp_x = box_3d_corner_map[j][0] * post_process_objects[i].dim0;
            float tmp_y = box_3d_corner_map[j][1] * post_process_objects[i].dim1;
            float tmp_z = box_3d_corner_map[j][2] * post_process_objects[i].dim2;

            float cos_value = cos(post_process_objects[i].yaw);
            float sin_value = sin(post_process_objects[i].yaw);

            float rotate_x = tmp_x * cos_value + tmp_z * sin_value + post_process_objects[i].x;
            float rotate_y = tmp_y + post_process_objects[i].y;
            float rotate_z = tmp_z * cos_value - tmp_x * sin_value + post_process_objects[i].z;

            float box3d_x = rotate_x * camera_k_waymo[0][0] + rotate_z * camera_k_waymo[0][2];
            float box3d_y = rotate_y * camera_k_waymo[1][1] + rotate_z * camera_k_waymo[1][2];
            float box3d_z = rotate_z;

            object.coo[j][0] = box3d_x / box3d_z;
            object.coo[j][1] = box3d_y / box3d_z;
        }
        box_3d_objects.push_back(object);
    }
}

void draw_box_3d_object(const char* image_file, const std::vector<box_3d_object> box_3d_objects)
{
    cv::Mat input = cv::imread(image_file);
    cv::Mat input_poly = input.clone();

    for (int i = 0; i < box_3d_objects.size(); ++i)
    {
        box_3d_object object = box_3d_objects[i];
        for (int j = 3; j >= 0; j--)
        {
            for (int k = 0; k < 4; ++k)
            {
                cv::line(input, cv::Point(object.coo[face_idx[j][k]][0], object.coo[face_idx[j][k]][1]),
                         cv::Point(object.coo[face_idx[j][(k + 1) % 4]][0], object.coo[face_idx[j][(k + 1) % 4]][1]),
                         cv::Scalar(0, 255, 0), 1, cv::LineTypes::LINE_AA);
            }
            if (j == 0)
            {
                // cv::Point poly_points[0][4];             // dimension can not be 0
                cv::Point poly_points[1][4];
                poly_points[0][0] = cv::Point(object.coo[face_idx[0][0]][0], object.coo[face_idx[0][0]][1]);
                poly_points[0][1] = cv::Point(object.coo[face_idx[0][1]][0], object.coo[face_idx[0][1]][1]);
                poly_points[0][2] = cv::Point(object.coo[face_idx[0][2]][0], object.coo[face_idx[0][2]][1]);
                poly_points[0][3] = cv::Point(object.coo[face_idx[0][3]][0], object.coo[face_idx[0][3]][1]);
                int npt[] = {4};
                const cv::Point* ppt[1] = {poly_points[0]};
                cv::fillPoly(input_poly, ppt, npt, 1, cv::Scalar(0, 0, 255));
            }
        }
    }
    cv::addWeighted(input, 0.8, input_poly, 0.2, 10, input);
    cv::imwrite("tengine_apollo_smoke_res.png", input);
}

void get_smoke_input_data(float* input_data, const char* image_file, const float* means, const float* scale)
{
    cv::Mat input = cv::imread(image_file);
    cv::resize(input, input, cv::Size(DEFAULT_IMG_W, DEFAULT_IMG_H), cv::INTER_LINEAR);
    for (int h = 0; h < DEFAULT_IMG_H; h++)
    {
        for (int w = 0; w < DEFAULT_IMG_W; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * DEFAULT_IMG_W * 3 + w * 3 + c;
                int out_index = c * DEFAULT_IMG_W * DEFAULT_IMG_H + h * DEFAULT_IMG_W + w;
                float tmp = ((float)input.data[in_index] / 255.f - means[c]) * scale[c];
                input_data[out_index] = tmp;
            }
        }
    }
}

int tengine_apollo_smoke(const char* model_file,
                         const char* image_file,
                         int img_h,
                         int img_w,
                         int loop_count,
                         int num_thread,
                         int affinity)
{
    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = affinity;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the shape, data buffer of input_tensor of the graph */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
    auto* input_data = (float*)malloc(img_size * sizeof(float));
    float means[3] = {0.485, 0.456, 0.406};
    float scales[3] = {1 / 0.229, 1 / 0.224, 1 / 0.225};

    get_smoke_input_data(input_data, image_file, means, scales);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data, img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < loop_count; i++)
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
    fprintf(stderr, "\nmodel file : %s\n", model_file);
    fprintf(stderr, "image file : %s\n", image_file);
    fprintf(stderr, "img_h, img_w, scale[3], mean[3] : %d %d , %.3f %.3f %.3f, %.1f %.1f %.1f\n", img_h, img_w,
            scales[0], scales[1], scales[2], means[0], means[1], means[2]);
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", loop_count,
            num_thread, total_time / loop_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    tensor_t hm_tensor = get_graph_output_tensor(graph, 0, 0);
    tensor_t reg_tensor = get_graph_output_tensor(graph, 1, 0);
    tensor_t hm_max_tensor = get_graph_output_tensor(graph, 2, 0);
    auto* hm_data = (float*)get_tensor_buffer(hm_tensor);
    auto* reg_data = (float*)get_tensor_buffer(reg_tensor);
    auto* hm_max_data = (float*)get_tensor_buffer(hm_max_tensor);

    int hm_dim[4];
    get_tensor_shape(hm_tensor, hm_dim, 4);
    int c = hm_dim[1], h = hm_dim[2], w = hm_dim[3];

    // 1. process hm message get object score and position
    std::vector<hm_process_object> hm_process_objects;
    process_hm_message(hm_process_objects, c, h, w, hm_max_data, hm_data);

    // 2. get regression data by hm position
    std::vector<reg_process_object> reg_process_objects;
    get_reg_data_object(hm_process_objects, reg_process_objects, h, w, reg_data);

    // 3. post process regression data
    std::vector<post_process_object> post_process_objects;
    post_process(hm_process_objects, reg_process_objects, post_process_objects);

    // 4. get object 8 corner points
    std::vector<box_3d_object> box_3d_objects;
    box_3d_process(post_process_objects, box_3d_objects);

    draw_box_3d_object(image_file, box_3d_objects);

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file]\n [-g img_h,img_w] [-s scale[0],scale[1],scale[2]] [-w "
        "mean[0],mean[1],mean[2]] [-r loop_count] [-t thread_count] [-a cpu_affinity]\n");
    fprintf(
        stderr,
        "\nmobilenet example: \n    ./classification -m /path/to/mobilenet.tmfile -i /path/to/img.jpg -g 224,224 -s "
        "0.017,0.017,0.017 -w 104.007,116.669,122.679\n");
}

int main(int argc, char* argv[])
{
    int loop_count = DEFAULT_LOOP_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    int cpu_affinity = DEFAULT_CPU_AFFINITY;
    char* model_file = NULL;
    char* image_file = NULL;
    float img_hw[2] = {0.f};
    int img_h = 0;
    int img_w = 0;

    int res;
    while ((res = getopt(argc, argv, "m:i:l:g:s:w:r:t:a:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'g':
            split(img_hw, optarg, ",");
            img_h = (int)img_hw[0];
            img_w = (int)img_hw[1];
            break;
        case 'r':
            loop_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'a':
            cpu_affinity = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default: break;
        }
    }

    /* check files */
    if (model_file == nullptr)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == nullptr)
    {
        fprintf(stderr, "Error: image file not specified!\n");
        show_usage();
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    if (img_h == 0)
    {
        img_h = DEFAULT_IMG_H;
        fprintf(stderr, "Image height not specified, use default %d\n", img_h);
    }

    if (img_w == 0)
    {
        img_w = DEFAULT_IMG_W;
        fprintf(stderr, "Image width not specified, use default  %d\n", img_w);
    }

    if (tengine_apollo_smoke(model_file, image_file, img_h, img_w, loop_count, num_thread, cpu_affinity) < 0)
        return -1;

    return 0;
}
