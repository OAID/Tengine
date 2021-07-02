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
 * Author: zylo117
 *
 * original model: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
 */

#include <stdlib.h>
#include <stdio.h>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_IMG_H 512
#define DEFAULT_IMG_W 512
#define DEFAULT_SCALE1 0.017124754f
#define DEFAULT_SCALE2 0.017507003f
#define DEFAULT_SCALE3 0.017429194f
#define DEFAULT_MEAN1 123.675
#define DEFAULT_MEAN2 116.280
#define DEFAULT_MEAN3 103.530
#define DEFAULT_LOOP_COUNT 1
#define DEFAULT_THREAD_COUNT 1
#define DEFAULT_CPU_AFFINITY 255


typedef struct Box
{
    int x0;
    int y0;
    int x1;
    int y1;
    int class_idx;
    float score;
} Box_t;


void qsort_descent_inplace(Box_t* boxes, int left, int right) {
    int i = left;
    int j = right;
    float p = boxes[(left + right) / 2].score;

    while (i <= j) {
        while (boxes[i].score > p)
            i++;

        while (boxes[j].score < p)
            j--;

        if (i <= j) {
            // swap
            Box_t tmp = boxes[i];
            boxes[i] = boxes[j];
            boxes[j] = tmp;

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(boxes, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(boxes, i, right);
        }
    }
}


int nms(const Box_t* boxes, const int num_boxes, int* suppressed, float nms_threshold) {
    int num_outputs = num_boxes;

    float* areas = malloc(num_boxes * sizeof(float));

    for (int i = 0; i < num_boxes; i++) {
        areas[i] = (float) ((boxes[i].x1 - boxes[i].x0) * (boxes[i].y1 - boxes[i].y0));
    }

    for (int i = 0; i < num_boxes; i++) {
        const Box_t a = boxes[i];

        if (suppressed[i] == 1)
            continue;

        for (int j = i + 1; j < num_boxes; j++) {
            const Box_t b = boxes[j];

            if (suppressed[j] == 1)
                continue;

            // iou
            float intersection = fmaxf(fminf(a.x1, b.x1) - fmaxf(a.x0, b.x0), 0) * fmaxf(fminf(a.y1, b.y1) - fmaxf(a.y0, b.y0), 0);
            float total_area = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - intersection;
            float iou = fmaxf(intersection / total_area, 0);

            if (iou > nms_threshold){
                suppressed[j] = 1;
                num_outputs--;
            } else{
                suppressed[j] = 0;
            }
        }
    }

    free(areas);
    return num_outputs;
}


float* arange(int start, int end, float stride) {
    int length = (int) ((float) ceilf((float) (end - start) / stride));
    float* result = malloc(length * sizeof(float));

    result[0] = (float) start;
    for (int i = 1; i < length; i++) {
        result[i] = result[i - 1] + stride;
    }
    return result;
}


void tile(const float* arr, int arr_length, int times, float offset,
            float* result, int arr_starts_from, int arr_stride) {
    int length = arr_length * times;

    if (result == NULL) {
        result = malloc(length * sizeof(float));
        arr_starts_from = 0;
    }

    for (int i = 0, j = 0; i < length; i++, j += arr_stride) {
        result[j + arr_starts_from] = arr[i % arr_length] + offset;
    }
}

void repeat(const float* arr, int arr_length, int times, float offset,
              float* result, int arr_starts_from, int arr_stride) {
    int length = arr_length * times;

    if (result == NULL) {
        result = malloc(length * sizeof(float));
        arr_starts_from = 0;
    }

    for (int i = 0, j = 0; i < length; i++, j += arr_stride) {
        result[j + arr_starts_from] = arr[i / times] + offset;
    }
}


int argmax(const float* arr, int arr_starts_from, int arr_length) {
    float max_value = arr[arr_starts_from];
    int max_idx = 0;
    for (int i = 1; i < arr_length; i++) {
        float this_value = arr[arr_starts_from + i];
        if (this_value > max_value) {
            max_value = this_value;
            max_idx = i;
        }
    }
    return max_idx;
}


int tengine_detect(const char* model_file, const char* image_file, int img_h, int img_w, const float* mean,
                     const float* scale, int loop_count, int num_thread, int affinity)
{
    /* setup network */
    const char* CLASSES_NAME[] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                 "fire hydrant", "", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                                 "cow", "elephant", "bear", "zebra", "giraffe", "", "backpack", "umbrella", "", "", "handbag", "tie",
                                 "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                                 "skateboard", "surfboard", "tennis racket", "bottle", "", "wine glass", "cup", "fork", "knife", "spoon",
                                 "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                                 "cake", "chair", "couch", "potted plant", "bed", "", "dining table", "", "", "toilet", "", "tv",
                                 "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                                 "refrigerator", "", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                                 "toothbrush"};

    int PYRAMID_LEVELS[] = {3, 4, 5, 6, 7};
    int STRIDES[] = {8, 16, 32, 64, 128};
    float SCALES[] = {
                (float) pow(2, 0.),
                (float) pow(2, 1. / 3.),
                (float) pow(2, 2. / 3.),
    };
    float RATIOS_X[] = {1.f, 1.4f, 0.7f};
    float RATIOS_Y[] = {1.f, 0.7f, 1.4f};
    float ANCHOR_SCALE = 4.f;
    float CONFIDENCE_THRESHOLD = 0.2f;
    float NMS_THRESHOLD = 0.2f;

    int num_levels = sizeof(PYRAMID_LEVELS) / sizeof(int);
    int num_scales = sizeof(SCALES) / sizeof(float);
    int num_ratios = sizeof(RATIOS_X) / sizeof(float);

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
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the shape, data buffer of input_tensor of the graph */
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
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    float means[3] = {mean[0], mean[1], mean[2]};
    float scales[3] = {scale[0], scale[1], scale[2]};
    image im = imread(image_file);
    image im_vis = copy_image(im);

    im = imread2caffe(im, img_w, img_h, means, scales);

    int raw_h = im.h;
    int raw_w = im.w;
    int resized_h, resized_w;
    float resize_scale;
    image resImg;
    if (raw_h > raw_w){
        resized_h = img_h;
        resized_w = (int) ((float) img_h / raw_h * raw_w);
        resImg = resize_image(im, resized_w, img_h);
        resize_scale = (float) raw_h / img_h;
    } else{
        resized_w = img_w;
        resized_h = (int) ((float) img_w / raw_w * raw_h);
        resImg = resize_image(im, img_w, resized_h);
        resize_scale = (float) raw_w / img_w;
    }
    free_image(im);

    image paddedImg = copyMaker(resImg, 0, img_h - resized_h, 0, img_w - resized_w, 0);
    free_image(resImg);

    memcpy(input_data, paddedImg.data, sizeof(float) * paddedImg.c * img_w * img_h);
    free_image(paddedImg);

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
            scale[0], scale[1], scale[2], mean[0], mean[1], mean[2]);
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", loop_count,
            num_thread, total_time / loop_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* get the result of classification */
    tensor_t output_tensor_regression = get_graph_output_tensor(graph, 0, 0);
    float* output_data_regression = ( float* )get_tensor_buffer(output_tensor_regression);
    int num_anchors = get_tensor_buffer_size(output_tensor_regression) / sizeof(float) / 4;

    tensor_t output_tensor_classification = get_graph_output_tensor(graph, 1, 0);
    float* output_data_classification = ( float* )get_tensor_buffer(output_tensor_classification);
    int num_classes = get_tensor_buffer_size(output_tensor_classification) / sizeof(float) / num_anchors;

    // postprocess
    // generate anchors
    float* anchors_x0 = malloc(num_anchors * sizeof(float));
    float* anchors_x1 = malloc(num_anchors * sizeof(float));
    float* anchors_y0 = malloc(num_anchors * sizeof(float));
    float* anchors_y1 = malloc(num_anchors * sizeof(float));

    int anchor_idx = 0;
    for (int stride_idx = 0; stride_idx < num_levels; stride_idx++) {
        int stride = STRIDES[stride_idx];
        float arange_stride = powf(2, (float) PYRAMID_LEVELS[stride_idx]);
        int length_x = (int) ceilf(((float) img_w - (float) stride / 2) / (float) arange_stride);
        int length_y = (int) ceilf(((float) img_h - (float) stride / 2) / (float) arange_stride);
        float* x = arange(stride / 2, img_w, arange_stride);
        float* y = arange(stride / 2, img_h, arange_stride);

        int start_idx = anchor_idx;
        int num_anchor_types = num_scales * num_ratios;
        for (int i = 0; i < num_scales; i++) {
            float anchor_scale = SCALES[i];
            float base_anchor_size = ANCHOR_SCALE * (float) stride * anchor_scale;

            for (int j = 0; j < num_ratios; j++) {
                float ratio_x = RATIOS_X[j];
                float ratio_y = RATIOS_Y[j];

                float anchor_size_x_2 = base_anchor_size * ratio_x / 2.f;
                float anchor_size_y_2 = base_anchor_size * ratio_y / 2.f;

                tile(x, length_x, length_y, -anchor_size_x_2, anchors_x0,
                     start_idx + i * num_scales + j, num_anchor_types);
                repeat(y, length_y, length_x, -anchor_size_y_2, anchors_y0,
                       start_idx + i * num_scales + j, num_anchor_types);
                tile(x, length_x, length_y, anchor_size_x_2, anchors_x1,
                     start_idx + i * num_scales + j, num_anchor_types);
                repeat(y, length_y, length_x, anchor_size_y_2, anchors_y1,
                       start_idx + i * num_scales + j, num_anchor_types);

                anchor_idx += (length_x * length_y);
            }
        }
        free(x);
        free(y);
    }

    // loop over anchors
    Box_t* proposals = malloc(sizeof(Box_t) * num_anchors);
    int num_proposals_over_threshold = 0;

#pragma omp parallel for num_threads(opt.num_thread)
    for (int i = 0; i < num_anchors; i++) {
        // loop over anchors

        // confidence
        int max_idx = argmax(output_data_classification, i * num_classes, num_classes);
        float max_score = output_data_classification[i * num_classes + max_idx];

        if (isinf(max_score) || max_score < CONFIDENCE_THRESHOLD){
            proposals[i].class_idx = -1;
            continue;
        }

        proposals[i].class_idx = max_idx;
        proposals[i].score = max_score;

        // box transform
        float ha = anchors_y1[i] - anchors_y0[i];
        float wa = anchors_x1[i] - anchors_x0[i];
        float y_center_a = (anchors_y1[i] + anchors_y0[i]) / 2;
        float x_center_a = (anchors_x1[i] + anchors_x0[i]) / 2;

        float w = expf(output_data_regression[i * 4 + 3]) * wa;
        float h = expf(output_data_regression[i * 4 + 2]) * ha;
        float y_center = output_data_regression[i * 4] * ha + y_center_a;
        float x_center = output_data_regression[i * 4 + 1] * wa + x_center_a;

        float ymin = y_center - h / 2;
        float xmin = x_center - w / 2;
        float ymax = y_center + h / 2;
        float xmax = x_center + w / 2;

        // scaling
        ymin *= resize_scale;
        xmin *= resize_scale;
        ymax *= resize_scale;
        xmax *= resize_scale;

        // clipping
        xmin = fmaxf(fminf(xmin, (float) (raw_w - 1)), 0.f);
        xmax = fmaxf(fminf(xmax, (float) (raw_w - 1)), 0.f);
        ymin = fmaxf(fminf(ymin, (float) (raw_h - 1)), 0.f);
        ymax = fmaxf(fminf(ymax, (float) (raw_h - 1)), 0.f);

        // area filtering
        float area = (xmax - xmin) * (ymax - ymin);
        if (area < 4){
            proposals[i].class_idx = -1;
            continue;
        }

        num_proposals_over_threshold++;

        proposals[i].x0 = (int) xmin;
        proposals[i].x1 = (int) xmax;
        proposals[i].y0 = (int) ymin;
        proposals[i].y1 = (int) ymax;
    }
    free(anchors_x0);
    free(anchors_x1);
    free(anchors_y0);
    free(anchors_y1);
    free(output_data_regression);
    free(output_data_classification);

    // filter boxes with confidence threshold
    Box_t* proposals_over_threshold = malloc(sizeof(Box_t) * num_proposals_over_threshold);
    int proposals_over_threshold_idx = 0;
    for (int i = 0; i < num_anchors; i++) {
        Box_t box = proposals[i];
        if(box.class_idx == -1)
            continue;
        proposals_over_threshold[proposals_over_threshold_idx] = box;
        proposals_over_threshold_idx++;
    }
    free(proposals);

    if (num_proposals_over_threshold > 0){
        // sort boxes
        qsort_descent_inplace(proposals_over_threshold, 0, num_proposals_over_threshold - 1);

        // nms
        int* suppressed = calloc(num_proposals_over_threshold, sizeof(int));
        int num_outputs = nms(proposals_over_threshold, num_proposals_over_threshold, suppressed, NMS_THRESHOLD);
        Box_t* proposals_after_nms = malloc(num_outputs * sizeof(Box_t));
        int proposals_after_nms_idx = 0;
        for(int i = 0; i < num_proposals_over_threshold; i++){
            Box_t box = proposals_over_threshold[i];
            if(suppressed[i] == 1)
                continue;
            proposals_after_nms[proposals_after_nms_idx] = box;
            proposals_after_nms_idx++;
        }
        free(suppressed);

        for (int i = 0; i < num_outputs; i++)
        {
            Box_t box = proposals_after_nms[i];
            draw_box(im_vis, box.x0, box.y0, box.x1, box.y1, 2, 125, 0, 125);
            fprintf(stderr, "%s\t:%.1f%%\n", CLASSES_NAME[box.class_idx], box.score * 100);
            fprintf(stderr, "BOX:( %d , %d ),( %d , %d )\n", box.x0, box.y0, box.x1, box.y1);
        }

        save_image(im_vis, "efficientdet_out");

        free(proposals_after_nms);
    }
    free(proposals_over_threshold);

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
        "\nefficientdet example: \n    ./classification -m /path/to/efficientdet.tmfile -i /path/to/img.jpg -g 512,512 -s "
        "0.017,0.017,0.017 -w 103.53,116.28,123.675\n");
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
    float mean[3] = {-1.f, -1.f, -1.f};
    float scale[3] = {0.f, 0.f, 0.f};

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
                img_h = ( int )img_hw[0];
                img_w = ( int )img_hw[1];
                break;
            case 's':
                split(scale, optarg, ",");
                break;
            case 'w':
                split(mean, optarg, ",");
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

    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        scale[0] = DEFAULT_SCALE1;
        scale[1] = DEFAULT_SCALE2;
        scale[2] = DEFAULT_SCALE3;
        fprintf(stderr, "Scale value not specified, use default  %.3f, %.3f, %.3f\n", scale[0], scale[1], scale[2]);
    }

    if (mean[0] == -1.0 || mean[1] == -1.0 || mean[2] == -1.0)
    {
        mean[0] = DEFAULT_MEAN1;
        mean[1] = DEFAULT_MEAN2;
        mean[2] = DEFAULT_MEAN3;
        fprintf(stderr, "Mean value not specified, use default   %.1f, %.1f, %.1f\n", mean[0], mean[1], mean[2]);
    }

    if (tengine_detect(model_file, image_file, img_h, img_w, mean, scale, loop_count, num_thread, cpu_affinity) < 0)
        return -1;

    return 0;
}
