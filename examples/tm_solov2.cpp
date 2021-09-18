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
 * Author: 774074168@qq.com
 * original model: https://github.com/WXinlong/SOLO
 */

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>
#include <map>
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "operator/prototype/convolution_param.h"

typedef int (*common_test)(graph_t, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w, int outc);
int create_input_node(graph_t graph, const char* node_name, int data_type, int layout, int n, int c, int h, int w, int dims_count = 4)
{
    if (0 == n) dims_count = 3;
    if (0 == c) dims_count = 2;
    if (0 == h) dims_count = 1;
    if (0 == w)
    {
        fprintf(stderr, "Dim of input node is not allowed. { n, c, h, w } = {%d, %d, %d, %d}.\n", n, c, h, w);
        return -1;
    }

    node_t node = create_graph_node(graph, node_name, "InputOp");
    if (NULL == node)
    {
        fprintf(stderr, "Create %d dims node(%s) failed. ", dims_count, node_name);
        return -1;
    }

    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);
    if (NULL == tensor)
    {
        release_graph_node(node);

        fprintf(stderr, "Create %d dims tensor for node(%s) failed. ", dims_count, node_name);

        return -1;
    }

    int ret = set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);
    if (0 != ret)
    {
        release_graph_tensor(tensor);
        release_graph_node(node);

        fprintf(stderr, "Set %d dims output tensor for node(%s) failed. ", dims_count, node_name);

        return -1;
    }

    switch (dims_count)
    {
    case 1:
    {
        int dims_array[1] = {w};
        set_tensor_shape(tensor, dims_array, dims_count);
        break;
    }
    case 2:
    {
        int dims_array[2] = {h, w};
        set_tensor_shape(tensor, dims_array, dims_count);
        break;
    }
    case 3:
    {
        if (TENGINE_LAYOUT_NCHW == layout)
        {
            int dims_array[3] = {c, h, w};
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }

        if (TENGINE_LAYOUT_NHWC == layout)
        {
            int dims_array[3] = {h, w, c};
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }
    }
    case 4:
    {
        if (TENGINE_LAYOUT_NCHW == layout)
        {
            int dims_array[4] = {n, c, h, w};
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }

        if (TENGINE_LAYOUT_NHWC == layout)
        {
            int dims_array[4] = {n, h, w, c};
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }
    }
    case 5:
    {
        if (TENGINE_LAYOUT_NCHW == layout)
        {
            int dims_array[5] = {1, n, c, h, w};
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }

        if (TENGINE_LAYOUT_NHWC == layout)
        {
            int dims_array[5] = {1, n, h, w, c};
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }
    }
    default:
        fprintf(stderr, "Cannot support %d dims tensor.\n", dims_count);
    }

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}
graph_t create_common_test_graph(const char* test_node_name, int data_type, int layout, int n, int c, int h, int w, int outc, common_test test_func, int dims_num = 4)
{
    graph_t graph = create_graph(NULL, NULL, NULL);
    if (NULL == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return NULL;
    }

    if (set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return NULL;
    }

    const char* input_name = "input_node";
    if (create_input_node(graph, input_name, data_type, layout, n, c, h, w, dims_num) < 0)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    if (test_func(graph, input_name, test_node_name, data_type, layout, n, c, h, w, outc) < 0)
    {
        fprintf(stderr, "create test node failed.\n");
        return NULL;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {test_node_name};

    if (set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return NULL;
    }

    if (set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return NULL;
    }

    return graph;
}

struct Object
{
    int cx;
    int cy;
    int label;
    float prob;
    cv::Mat mask;
};

static inline float intersection_area(const Object& a, const Object& b, int img_w, int img_h)
{
    float area = 0.f;
    for (int y = 0; y < img_h; y = y + 4)
    {
        for (int x = 0; x < img_w; x = x + 4)
        {
            const uchar* mp1 = a.mask.ptr(y);
            const uchar* mp2 = b.mask.ptr(y);
            if (mp1[x] == 255 && mp2[x] == 255) area += 1.f;
        }
    }
    return area;
}

static inline float area(const Object& a, int img_w, int img_h)
{
    float area = 0.f;
    for (int y = 0; y < img_h; y = y + 4)
    {
        for (int x = 0; x < img_w; x = x + 4)
        {
            const uchar* mp = a.mask.ptr(y);
            if (mp[x] == 255) area += 1.f;
        }
    }
    return area;
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_segs(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, int img_w, int img_h)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = area(objects[i], img_w, img_h);
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b, img_w, img_h);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void kernel_pick(const float* cate_pred, int w, int h, std::vector<int>& picked, int num_class, float cate_thresh)
{
    for (int q = 0; q < num_class; q++)
    {
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                int index = i * w + j;
                float cate_score = cate_pred[q * h * w + index];

                if (cate_score < cate_thresh)
                {
                    continue;
                }
                else
                    picked.push_back(index);
            }
        }
    }
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

void get_input_data(const char* image_file, float* input_data, int letterbox_rows, int letterbox_cols, const float* mean, const float* scale, int& wpad, int& hpad)
{
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / img.rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));

    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_32FC3, cv::Scalar(0, 0, 0));
    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;
    hpad = letterbox_rows - resize_rows;
    wpad = letterbox_cols - resize_cols;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(114.f, 114.f, 114.f));

    float* img_data = (float*)img_new.data;

    /* nhwc to nchw */
    for (int h = 0; h < letterbox_rows; h++)
    {
        for (int w = 0; w < letterbox_cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * letterbox_cols * 3 + w * 3 + c;
                int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                input_data[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }
}

int create_test_conv_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w, int outc)
{
    (void)layout;
    (void)n;
    (void)c;
    (void)h;
    (void)w;
    (void)outc;

    /* create the test node */
    struct node* test_node = (struct node*)create_graph_node(graph, node_name, "Convolution");
    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if (nullptr == input_tensor)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    /* create the sub node to product another input tensors which the test node is needed, such as weight/bias/slope tensor. */
    /* weight */
    node_t weight_node = create_graph_node(graph, "weight", "Const");
    tensor_t weight_tensor = create_graph_tensor(graph, "weight", TENGINE_DT_FP32);
    set_node_output_tensor(weight_node, 0, weight_tensor, TENSOR_TYPE_CONST);
    int weight_dims[4] = {outc, c, 1, 1}; // channel num
    set_tensor_shape(weight_tensor, weight_dims, 4);

    /* bias */
    node_t bias_node = create_graph_node(graph, "bias", "Const");
    tensor_t bias_tensor = create_graph_tensor(graph, "bias", TENGINE_DT_FP32);
    set_node_output_tensor(bias_node, 0, bias_tensor, TENSOR_TYPE_CONST);
    int bias_dims[1] = {outc}; // channel num
    set_tensor_shape(bias_tensor, bias_dims, 1);

    /* input tensors of test node */
    set_node_input_tensor(test_node, 0, input_tensor);
    set_node_input_tensor(test_node, 1, weight_tensor);
    set_node_input_tensor(test_node, 2, bias_tensor);

    /* output tensors of test node */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    /* set params */
    struct conv_param* conv_param = (struct conv_param*)(struct node*)test_node->op.param_mem;

    conv_param->kernel_h = 1;
    conv_param->kernel_w = 1;
    conv_param->stride_h = 1;
    conv_param->stride_w = 1;
    conv_param->pad_h0 = 0;
    conv_param->pad_h1 = 0;
    conv_param->pad_w0 = 0;
    conv_param->pad_w1 = 0;
    conv_param->dilation_h = 1;
    conv_param->dilation_w = 1;
    conv_param->input_channel = c;
    conv_param->output_channel = outc;
    conv_param->group = 1;
    conv_param->activation = -1;

    return 0;
}
static int ins_decode(float* kernel_pred, float* feature_pred,
                      std::vector<int>& kernel_picked, std::map<int, int>& kernel_map, std::vector<std::vector<float> >& ins_pred, int c_in)
{
    std::set<int> kernel_pick_set;
    kernel_pick_set.insert(kernel_picked.begin(), kernel_picked.end());
    int c_out = kernel_pick_set.size();
    int ret = 0;
    if (c_out > 0)
    {
        std::vector<float> bias_data(c_out, 0);
        //init graph
        ret = init_tengine();
        if (0 != ret)
            fprintf(stderr, "Tengine init failed.\n");

        // create
        graph_t graph = create_common_test_graph("conv", TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 1, c_in, 112, 112, c_out, &create_test_conv_node);
        if (nullptr == graph)
            return -1;

        //set_log_level(LOG_INFO);
        //dump_graph(graph);

        /* fill test data */
        // set quantize params
        struct tensor* input_tensor = (struct tensor*)get_graph_tensor(graph, "input_node");
        struct tensor* weight_tensor = (struct tensor*)get_graph_tensor(graph, "weight");
        struct tensor* bias_tensor = (struct tensor*)get_graph_tensor(graph, "bias");
        struct tensor* output_tensor = (struct tensor*)get_graph_tensor(graph, "conv");

        // set input data
        set_tensor_buffer(input_tensor, feature_pred, c_in * 112 * 112 * sizeof(float));

        std::vector<float> weights(c_in * c_out);
        std::set<int>::iterator pick_c;
        int count_c = 0;
        for (pick_c = kernel_pick_set.begin(); pick_c != kernel_pick_set.end(); pick_c++)
        {
            kernel_map[*pick_c] = count_c;
            for (int j = 0; j < c_in; j++)
            {
                weights[count_c * c_in + j] = kernel_pred[c_in * (*pick_c) + j];
            }
            count_c++;
        }
        // set weight data
        set_tensor_buffer(weight_tensor, weights.data(), c_in * c_out * sizeof(float));

        // set bias data
        set_tensor_buffer(bias_tensor, bias_data.data(), c_out * sizeof(float));

        // graph run
        if (prerun_graph(graph) < 0)
        {
            fprintf(stderr, "Pre-run graph failed.\n");
            return -1;
        }

        if (0 != run_graph(graph, 1))
        {
            fprintf(stderr, "Run graph error.\n");
            postrun_graph(graph);
            destroy_graph(graph);
            release_tengine();
            return -1;
        }

        /* get output*/
        int output_size = output_tensor->elem_num;
        float* output_fp32 = (float*)output_tensor->data;

        for (int i = 0; i < output_tensor->dims[1]; i++)
        {
            std::vector<float> tmp;
            for (int j = 0; j < output_tensor->dims[2] * output_tensor->dims[3]; j++)
                tmp.push_back(output_fp32[i * output_tensor->dims[2] * output_tensor->dims[3] + j]);
            ins_pred.push_back(tmp);
        }
        // exit
        postrun_graph(graph);
        destroy_graph(graph);
        release_tengine();
    }

    return 0;
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void generate_res(float* cate_pred, std::vector<std::vector<float> > ins_pred, std::map<int, int>& kernel_map,
                  std::vector<std::vector<Object> >& objects, float cate_thresh,
                  float conf_thresh, int img_w, int img_h, int num_class, float stride, int wpad, int hpad,
                  int cate_pred_w, int cate_pred_h, int cate_pred_c)
{
    int w = cate_pred_w;
    int h = cate_pred_h;
    int w_ins = 112;
    int h_ins = 112;
    for (int q = 0; q < num_class; q++)
    {
        const float* cate_ptr = cate_pred + q * w * h;
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                int index = i * w + j;
                float cate_socre = cate_ptr[index];
                if (cate_socre < cate_thresh)
                {
                    continue;
                }
                const float* ins_ptr = ins_pred[kernel_map[index]].data();
                cv::Mat mask(h_ins, w_ins, CV_32FC1);
                float sum_mask = 0.f;
                int count_mask = 0;
                {
                    mask = cv::Scalar(0.f);
                    float* mp = (float*)mask.data;
                    for (int m = 0; m < w_ins * h_ins; m++)
                    {
                        float mask_score = sigmoid(ins_ptr[m]);

                        if (mask_score > 0.5)
                        {
                            mp[m] = mask_score;
                            sum_mask += mask_score;
                            count_mask++;
                        }
                    }
                }
                if (count_mask < stride)
                {
                    continue;
                }
                float mask_score = sum_mask / (float(count_mask) + 1e-6);

                float socre = mask_score * cate_socre;

                if (socre < conf_thresh)
                {
                    continue;
                }
                cv::Mat mask_cut;
                cv::Rect rect(wpad / 8, hpad / 8, w_ins - wpad / 4, h_ins - hpad / 4);
                mask_cut = mask(rect);
                cv::Mat mask2;
                cv::resize(mask_cut, mask2, cv::Size(img_w, img_h));
                Object obj;
                obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
                float sum_mask_y = 0.f;
                float sum_mask_x = 0.f;
                int area = 0;
                {
                    obj.mask = cv::Scalar(0);
                    for (int y = 0; y < img_h; y++)
                    {
                        const float* mp2 = mask2.ptr<const float>(y);
                        uchar* bmp = obj.mask.ptr<uchar>(y);
                        for (int x = 0; x < img_w; x++)
                        {
                            if (mp2[x] > 0.5f)
                            {
                                bmp[x] = 255;
                                sum_mask_y += (float)y;
                                sum_mask_x += (float)x;
                                area++;
                            }
                            else
                                bmp[x] = 0;
                        }
                    }
                }

                if (area < 100) continue;

                obj.cx = int(sum_mask_x / area);
                obj.cy = int(sum_mask_y / area);
                obj.label = q + 1;
                obj.prob = socre;
                objects[q].push_back(obj);
            }
        }
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, const char* save_path)
{
    static const char* class_names[] = {"background",
                                        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic light", "fire hydrant",
                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                        "baseball glove", "skateboard", "surfboard", "tennis racket",
                                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                        "scissors", "teddy bear", "hair drier", "toothbrush"};

    static const unsigned char colors[81][3] = {
        {56, 0, 255},
        {226, 255, 0},
        {0, 94, 255},
        {0, 37, 255},
        {0, 255, 94},
        {255, 226, 0},
        {0, 18, 255},
        {255, 151, 0},
        {170, 0, 255},
        {0, 255, 56},
        {255, 0, 75},
        {0, 75, 255},
        {0, 255, 169},
        {255, 0, 207},
        {75, 255, 0},
        {207, 0, 255},
        {37, 0, 255},
        {0, 207, 255},
        {94, 0, 255},
        {0, 255, 113},
        {255, 18, 0},
        {255, 0, 56},
        {18, 0, 255},
        {0, 255, 226},
        {170, 255, 0},
        {255, 0, 245},
        {151, 255, 0},
        {132, 255, 0},
        {75, 0, 255},
        {151, 0, 255},
        {0, 151, 255},
        {132, 0, 255},
        {0, 255, 245},
        {255, 132, 0},
        {226, 0, 255},
        {255, 37, 0},
        {207, 255, 0},
        {0, 255, 207},
        {94, 255, 0},
        {0, 226, 255},
        {56, 255, 0},
        {255, 94, 0},
        {255, 113, 0},
        {0, 132, 255},
        {255, 0, 132},
        {255, 170, 0},
        {255, 0, 188},
        {113, 255, 0},
        {245, 0, 255},
        {113, 0, 255},
        {255, 188, 0},
        {0, 113, 255},
        {255, 0, 0},
        {0, 56, 255},
        {255, 0, 113},
        {0, 255, 188},
        {255, 0, 94},
        {255, 0, 18},
        {18, 255, 0},
        {0, 255, 132},
        {0, 188, 255},
        {0, 245, 255},
        {0, 169, 255},
        {37, 255, 0},
        {255, 0, 151},
        {188, 0, 255},
        {0, 255, 37},
        {0, 255, 0},
        {255, 0, 170},
        {255, 0, 37},
        {255, 75, 0},
        {0, 0, 255},
        {255, 207, 0},
        {255, 0, 226},
        {255, 245, 0},
        {188, 255, 0},
        {0, 255, 18},
        {0, 255, 75},
        {0, 255, 151},
        {255, 56, 0},
        {245, 255, 0}};

    cv::Mat image = bgr.clone();

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        if (obj.prob < 0.15)
            continue;

        fprintf(stderr, "%d = %.5f at %.2d %.2d\n", obj.label, obj.prob,
                obj.cx, obj.cy);

        const unsigned char* color = colors[color_index % 81];
        color_index++;

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.cx;
        int y = obj.cy;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        // draw mask
        for (int y = 0; y < image.rows; y++)
        {
            const uchar* mp = obj.mask.ptr(y);
            uchar* p = image.ptr(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (mp[x] == 255)
                {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }

    cv::imwrite(save_path, image);
}
int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;

    int img_c = 3;
    const float mean[3] = {123.68f, 116.78f, 103.94f};
    const float scale[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    // allow none square letterbox, set default letterbox size
    int letterbox_rows = 448;
    int letterbox_cols = 448;

    int repeat_count = 1;
    int num_thread = 1;

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
            repeat_count = std::strtoul(optarg, nullptr, 10);
            break;
        case 't':
            num_thread = std::strtoul(optarg, nullptr, 10);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr == image_file)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    cv::Mat img = cv::imread(image_file, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_file);
        return -1;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    int img_size = letterbox_rows * letterbox_cols * img_c;
    int dims[] = {1, 3, int(letterbox_rows), int(letterbox_cols)};
    int dims3[] = {1, 2, int(letterbox_rows / 8), int(letterbox_cols / 8)};
    int dims4[] = {1, 2, int(letterbox_rows / 16), int(letterbox_cols / 16)};
    int dims5[] = {1, 2, int(letterbox_rows / 32), int(letterbox_cols / 32)};
    std::vector<float> input_data(img_size);
    std::vector<float> input_data3(2 * 56 * 56);
    std::vector<float> input_data4(2 * 28 * 28);
    std::vector<float> input_data5(2 * 14 * 14);

    tensor_t input_tensor = get_graph_tensor(graph, "input");
    tensor_t p3_input_tensor = get_graph_tensor(graph, "p3_input");
    tensor_t p4_input_tensor = get_graph_tensor(graph, "p4_input");
    tensor_t p5_input_tensor = get_graph_tensor(graph, "p5_input");

    if (input_tensor == nullptr || p3_input_tensor == nullptr || p4_input_tensor == nullptr || p5_input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0 || set_tensor_shape(p3_input_tensor, dims3, 4) < 0 || set_tensor_shape(p4_input_tensor, dims4, 4) < 0 || set_tensor_shape(p5_input_tensor, dims5, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size * 4) < 0 || set_tensor_buffer(p3_input_tensor, input_data3.data(), 2 * 56 * 56 * 4) < 0 || set_tensor_buffer(p4_input_tensor, input_data4.data(), 2 * 28 * 28 * 4) < 0 || set_tensor_buffer(p5_input_tensor, input_data5.data(), 2 * 14 * 14 * 4) < 0)
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
    int wpad, hpad;
    /* prepare process input data, set the data mem to input tensor */
    get_input_data(image_file, input_data.data(), letterbox_rows, letterbox_cols, mean, scale, wpad, hpad);

    int pw = int(letterbox_cols / 8);
    int ph = int(letterbox_rows / 8);
    float step_h = 2.f / (ph - 1);
    float step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++)
    {
        for (int w = 0; w < pw; w++)
        {
            input_data3[0 + h * pw + w] = -1.f + step_w * (float)w;
            input_data3[ph * pw + h * pw + w] = -1.f + step_h * (float)h;
        }
    }
    pw = int(letterbox_cols / 16);
    ph = int(letterbox_rows / 16);
    step_h = 2.f / (ph - 1);
    step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++)
    {
        for (int w = 0; w < pw; w++)
        {
            input_data4[0 + h * pw + w] = -1.f + step_w * (float)w;
            input_data4[ph * pw + h * pw + w] = -1.f + step_h * (float)h;
        }
    }
    pw = int(letterbox_cols / 32);
    ph = int(letterbox_rows / 32);
    step_h = 2.f / (ph - 1);
    step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++)
    {
        for (int w = 0; w < pw; w++)
        {
            input_data5[0 + h * pw + w] = -1.f + step_w * (float)w;
            input_data5[ph * pw + h * pw + w] = -1.f + step_h * (float)h;
        }
    }
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
        min_time = (std::min)(min_time, cur);
        max_time = (std::max)(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    tensor_t feature_pred = get_graph_tensor(graph, "feature_pred");
    tensor_t cate_pred1 = get_graph_tensor(graph, "cate_pred1");
    tensor_t cate_pred2 = get_graph_tensor(graph, "cate_pred2");
    tensor_t cate_pred3 = get_graph_tensor(graph, "cate_pred3");
    tensor_t cate_pred4 = get_graph_tensor(graph, "cate_pred4");
    tensor_t cate_pred5 = get_graph_tensor(graph, "cate_pred5");
    tensor_t kernel_pred1 = get_graph_tensor(graph, "kernel_pred1");
    tensor_t kernel_pred2 = get_graph_tensor(graph, "kernel_pred2");
    tensor_t kernel_pred3 = get_graph_tensor(graph, "kernel_pred3");
    tensor_t kernel_pred4 = get_graph_tensor(graph, "kernel_pred4");
    tensor_t kernel_pred5 = get_graph_tensor(graph, "kernel_pred5");
    float* feature_pred_data = (float*)get_tensor_buffer(feature_pred);

    float* cate_pred1_data = (float*)get_tensor_buffer(cate_pred1);
    float* cate_pred2_data = (float*)get_tensor_buffer(cate_pred2);
    float* cate_pred3_data = (float*)get_tensor_buffer(cate_pred3);
    float* cate_pred4_data = (float*)get_tensor_buffer(cate_pred4);
    float* cate_pred5_data = (float*)get_tensor_buffer(cate_pred5);
    float* kernel_pred1_data = (float*)get_tensor_buffer(kernel_pred1);
    float* kernel_pred2_data = (float*)get_tensor_buffer(kernel_pred2);
    float* kernel_pred3_data = (float*)get_tensor_buffer(kernel_pred3);
    float* kernel_pred4_data = (float*)get_tensor_buffer(kernel_pred4);
    float* kernel_pred5_data = (float*)get_tensor_buffer(kernel_pred5);

    int tensor_dims1[] = {0, 0, 0, 0};
    int tensor_dims2[] = {0, 0, 0, 0};
    int tensor_dims3[] = {0, 0, 0, 0};
    int tensor_dims4[] = {0, 0, 0, 0};
    int tensor_dims5[] = {0, 0, 0, 0};
    get_tensor_shape(cate_pred1, tensor_dims1, 4);
    get_tensor_shape(cate_pred2, tensor_dims2, 4);
    get_tensor_shape(cate_pred3, tensor_dims3, 4);
    get_tensor_shape(cate_pred4, tensor_dims4, 4);
    get_tensor_shape(cate_pred5, tensor_dims5, 4);

    const int target_size = 448;
    const float cate_thresh = 0.3f;
    const float confidence_thresh = 0.3f;
    const float nms_threshold = 0.3f;
    const int keep_top_k = 200;

    int num_class = tensor_dims1[1];
    std::vector<int> kernel_picked1, kernel_picked2, kernel_picked3, kernel_picked4, kernel_picked5;
    kernel_pick(cate_pred1_data, tensor_dims1[2], tensor_dims1[3], kernel_picked1, num_class, cate_thresh);
    kernel_pick(cate_pred2_data, tensor_dims2[2], tensor_dims2[3], kernel_picked2, num_class, cate_thresh);
    kernel_pick(cate_pred3_data, tensor_dims3[2], tensor_dims3[3], kernel_picked3, num_class, cate_thresh);
    kernel_pick(cate_pred4_data, tensor_dims4[2], tensor_dims4[3], kernel_picked4, num_class, cate_thresh);
    kernel_pick(cate_pred5_data, tensor_dims5[2], tensor_dims5[3], kernel_picked5, num_class, cate_thresh);

    int feature_pred_tensor_dim[] = {0, 0, 0, 0};
    get_tensor_shape(feature_pred, feature_pred_tensor_dim, 4);
    int c_in = feature_pred_tensor_dim[1];
    std::map<int, int> kernel_map1, kernel_map2, kernel_map3, kernel_map4, kernel_map5;
    std::vector<std::vector<float> > ins_pred1, ins_pred2, ins_pred3, ins_pred4, ins_pred5;

    ins_decode(kernel_pred1_data, feature_pred_data, kernel_picked1, kernel_map1, ins_pred1, c_in);
    ins_decode(kernel_pred2_data, feature_pred_data, kernel_picked2, kernel_map2, ins_pred2, c_in);
    ins_decode(kernel_pred3_data, feature_pred_data, kernel_picked3, kernel_map3, ins_pred3, c_in);
    ins_decode(kernel_pred4_data, feature_pred_data, kernel_picked4, kernel_map4, ins_pred4, c_in);
    ins_decode(kernel_pred5_data, feature_pred_data, kernel_picked5, kernel_map5, ins_pred5, c_in);

    std::vector<std::vector<Object> > class_candidates;
    class_candidates.resize(num_class);
    generate_res(cate_pred1_data, ins_pred1, kernel_map1, class_candidates, cate_thresh, confidence_thresh, img.cols, img.rows,
                 num_class, 8.f, wpad, hpad, tensor_dims1[3], tensor_dims1[2], tensor_dims1[1]);
    generate_res(cate_pred2_data, ins_pred2, kernel_map2, class_candidates, cate_thresh, confidence_thresh, img.cols, img.rows,
                 num_class, 8.f, wpad, hpad, tensor_dims2[3], tensor_dims2[2], tensor_dims2[1]);
    generate_res(cate_pred3_data, ins_pred3, kernel_map3, class_candidates, cate_thresh, confidence_thresh, img.cols, img.rows,
                 num_class, 16.f, wpad, hpad, tensor_dims3[3], tensor_dims3[2], tensor_dims3[1]);
    generate_res(cate_pred4_data, ins_pred4, kernel_map4, class_candidates, cate_thresh, confidence_thresh, img.cols, img.rows,
                 num_class, 32.f, wpad, hpad, tensor_dims4[3], tensor_dims4[2], tensor_dims4[1]);
    generate_res(cate_pred5_data, ins_pred5, kernel_map5, class_candidates, cate_thresh, confidence_thresh, img.cols, img.rows,
                 num_class, 32.f, wpad, hpad, tensor_dims5[3], tensor_dims5[2], tensor_dims5[1]);

    std::vector<Object> objects;
    objects.clear();
    for (int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidates = class_candidates[i];
        qsort_descent_inplace(candidates);
        std::vector<int> picked;
        nms_sorted_segs(candidates, picked, nms_threshold, img.cols, img.rows);
        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            objects.push_back(candidates[z]);
        }
    }
    qsort_descent_inplace(objects);
    // keep_top_k
    if (keep_top_k < (int)objects.size())
    {
        objects.resize(keep_top_k);
    }

    draw_objects(img, objects, "solov2_result.jpg");
    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}
