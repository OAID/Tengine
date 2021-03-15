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

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <vector>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include <sys/time.h>
#include "common.hpp"

#define DEF_MODEL "models/faster_rcnn.tmfile"
#define DEF_IMAGE "images/ssd_dog.jpg"
#define DEFAULT_REPEAT_CNT 1

typedef struct abox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_idx;
    bool operator<(const abox& tmp) const
    {
        return score < tmp.score;
    }
} abox;

void bbox_tranform_inv(float* local_anchors, float** boxs_delta, int num_roi, int imgw, int imgh)
{
    for(int i = 0; i < num_roi; i++)
    {
        double pred_ctr_x, pred_ctr_y, src_ctr_x, src_ctr_y;
        double dst_ctr_x, dst_ctr_y, dst_scl_x, dst_scl_y;
        double src_w, src_h, pred_w, pred_h;
        float* anchor = local_anchors + i * 4;
        src_w = anchor[2] - anchor[0] + 1;
        src_h = anchor[3] - anchor[1] + 1;
        src_ctr_x = anchor[0] + 0.5 * src_w;
        src_ctr_y = anchor[1] + 0.5 * src_h;

        dst_ctr_x = boxs_delta[i][0];
        dst_ctr_y = boxs_delta[i][1];
        dst_scl_x = boxs_delta[i][2];
        dst_scl_y = boxs_delta[i][3];
        pred_ctr_x = dst_ctr_x * src_w + src_ctr_x;
        pred_ctr_y = dst_ctr_y * src_h + src_ctr_y;
        pred_w = exp(dst_scl_x) * src_w;
        pred_h = exp(dst_scl_y) * src_h;

        boxs_delta[i][0] = pred_ctr_x - 0.5 * pred_w;
        if(boxs_delta[i][0] < 0)
            boxs_delta[i][0] = 0;
        if(boxs_delta[i][0] > imgw)
            boxs_delta[i][0] = imgw;

        boxs_delta[i][1] = pred_ctr_y - 0.5 * pred_h;
        if(boxs_delta[i][1] < 0)
            boxs_delta[i][1] = 0;
        if(boxs_delta[i][1] > imgh)
            boxs_delta[i][1] = imgh;

        boxs_delta[i][2] = pred_ctr_x + 0.5 * pred_w;
        if(boxs_delta[i][2] < 0)
            boxs_delta[i][2] = 0;
        if(boxs_delta[i][2] > imgw)
            boxs_delta[i][2] = imgw;

        boxs_delta[i][3] = pred_ctr_y + 0.5 * pred_h;
        if(boxs_delta[i][3] < 0)
            boxs_delta[i][3] = 0;
        if(boxs_delta[i][3] > imgh)
            boxs_delta[i][3] = imgh;
    }
}

void nms(std::vector<abox>& input_boxes, float nms_thresh)
{
    std::vector<float> vArea(input_boxes.size());
    for(int i = 0; i < ( int )input_boxes.size(); ++i)
    {
        vArea[i] =
            (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for(int i = 0; i < ( int )input_boxes.size(); ++i)
    {
        for(int j = i + 1; j < ( int )input_boxes.size();)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if(ovr >= nms_thresh)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

void draw_detections(std::string& image_file, std::string& save_name, std::vector<abox>& boxes)
{
    const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                 "sofa",       "train",     "tvmonitor"};
    image im = imread(image_file.c_str());

    for(int b = 0; b < ( int )boxes.size(); b++)
    {
        abox box = boxes[b];

        printf("%s\t: %.2f %%\n", class_names[box.class_idx], box.score * 100);
        printf("BOX:( %d , %d ),( %d , %d )\n", (int)box.x1, (int)box.y1, (int)box.x2, (int)box.y2);

        std::ostringstream score_str;
        score_str << box.score * 100;
        std::string labelstr = std::string(class_names[box.class_idx]);

        put_label(im, labelstr.c_str(), 0.02, box.x1, box.y1, 255, 255, 125);

        draw_box(im, box.x1, box.y1, box.x2, box.y2, 2, 0, 0, 0);
    }
    save_image(im, "tengine_example_out");
    free_image(im);
    std::cout << "======================================\n";
    std::cout << "[DETECTED IMAGE SAVED]:\t"
              << "Faster_Rcnn"
              << "\n";
    std::cout << "======================================\n";
}

int main(int argc, char* argv[])
{
    int ret = -1;
    int repeat_count = DEFAULT_REPEAT_CNT;
    const std::string root_path = get_root_path();
    std::string model_file;
    std::string image_file;
    std::string save_name = "save.jpg";

    int res;
    while((res = getopt(argc, argv, "p:m:i:")) != -1)
    {
        switch(res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            default:
                break;
        }
    }

    // init tengine
    if(init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }
    if(request_tengine_version("0.9") != 1)
    {
        std::cout << " request tengine version failed\n";
        return 1;
    }

    // load model
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
    // check file
    if((!check_file_exist(model_file) or !check_file_exist(image_file)))
    {
        return 1;
    }
    // load model & create graph
    graph_t graph = create_graph(nullptr, "tengine", model_file.c_str());

    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    int val = 1;
    set_graph_attr(graph, "low_mem_mode", &val, sizeof(val));

    image im = imread(image_file.c_str());
    if(im.data == 0)
        std::cerr << "Open pic Error: " << image_file << "\n";

    // preprocess img
    const int INPUT_SIZE_LONG = 500;
    const int INPUT_SIZE_NARROW = 300;

    int max_side = std::max(im.h, im.w);
    int min_side = std::min(im.h, im.w);

    float max_side_scale = float(max_side) / float(INPUT_SIZE_LONG);
    float min_side_scale = float(min_side) / float(INPUT_SIZE_NARROW);

    float max_scale = std::max(max_side_scale, min_side_scale);

    // im_info
    float img_scale = 1.f / max_scale;
    int height = int(im.h * img_scale);
    int width = int(im.w * img_scale);
    float im_info[3];
    im_info[0] = height;
    im_info[1] = width;
    im_info[2] = img_scale;
    //
    int hw = height * width;
    int img_size = hw * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);
    int img_w = width;
    int img_h = height;
    float mean[3] = {102.9801, 115.9465, 122.7717};


    float scales[3] = {1, 1, 1};
    free_image(im);
    image img = imread(image_file.c_str(), img_w, img_h, mean, scales, CAFFE);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h); 
    free_image(img);
    // std::cout<<"height width scale"<<height<<","<<width<<","<<img_scale<<"\n";
    // set input and output node
    const char* input_tensor_names[] = {"data", "im_info"};

    tensor_t input_tensor1 = get_graph_tensor(graph, input_tensor_names[0]);
    int dims[] = {1, 3, height, width};
    set_tensor_shape(input_tensor1, dims, 4);
    set_tensor_buffer(input_tensor1, input_data, img_size * 4);

    tensor_t input_tensor2 = get_graph_tensor(graph, input_tensor_names[1]);
    int dims1[] = {1, 3, 1, 1};
    set_tensor_shape(input_tensor2, dims1, 4);
    set_tensor_buffer(input_tensor2, im_info, 3 * 4);

    /* prerun the graph */
    struct options opt;
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    if(std::getenv("NumThreadLite"))
        opt.num_thread = atoi(std::getenv("NumThreadLite"));
    if(std::getenv("NumClusterLite"))
        opt.cluster = atoi(std::getenv("NumClusterLite"));
    if(std::getenv("DataPrecision"))
        opt.precision = atoi(std::getenv("DataPrecision"));
    if(std::getenv("REPEAT"))
        repeat_count = atoi(std::getenv("REPEAT"));
    
    std::cout<<"Number Thread  : [" << opt.num_thread <<"], use export NumThreadLite=1/2/4 set\n";
    std::cout<<"CPU Cluster    : [" << opt.cluster <<"], use export NumClusterLite=0/1/2/3 set\n";
    std::cout<<"Data Precision : [" << opt.precision <<"], use export DataPrecision=0/1/2/3 set\n";
    std::cout<<"Number Repeat  : [" << repeat_count <<"], use export REPEAT=10/100/1000 set\n";

    if(prerun_graph_multithread(graph, opt) < 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    struct timeval t0, t1;
    float total_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        ret = run_graph(graph, 1);
        if(ret != 0)
        {
            std::cout << "Run graph failed, errno: " << get_tengine_errno() << "\n";
            return 1;
        }
        gettimeofday(&t1, NULL);

        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
    }
    std::cout << "--------------------------------------\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n";


    tensor_t tensor = get_graph_tensor(graph, "rois");    //[1,num_roi,4,1]
    float* rois_data = ( float* )get_tensor_buffer(tensor);
    int rois_size = get_tensor_buffer_size(tensor) / sizeof(float);
    int roi_shape[4];
    get_tensor_shape(tensor, roi_shape, 4);
    int num_roi = roi_shape[1];
    for(int i = 0; i < 4 * num_roi; i++)
    {
        rois_data[i] /= img_scale;
    }

    tensor = get_graph_tensor(graph, "bbox_pred");    //[num_roi,21*4,1,1]
    float* bbox_delt_data = ( float* )get_tensor_buffer(tensor);
    int bbox_size = get_tensor_buffer_size(tensor) / sizeof(float);
    tensor = get_graph_tensor(graph, "cls_prob");    //[num_roi,21,1,1]
    float* score = ( float* )get_tensor_buffer(tensor);
    int cls_size = get_tensor_buffer_size(tensor) / sizeof(float);

    int cls_shape[4];
    get_tensor_shape(tensor, cls_shape, 4);
    int num_class = cls_shape[1];
    int chw = num_class * 4;
    std::vector<abox> all_boxes;
    float CONF_THRESH = 0.75;
    float NMS_THRESH = 0.3;
    float** bbox_delt = ( float** )calloc(num_roi, sizeof(float*));
    for(int k = 0; k < num_roi; k++)
        bbox_delt[k] = ( float* )calloc(4, sizeof(float*));
    for(int i = 1; i < num_class; i++)
    {
        for(int j = 0; j < num_roi; j++)
        {
            bbox_delt[j][0] = bbox_delt_data[j * chw + i * 4];
            bbox_delt[j][1] = bbox_delt_data[j * chw + i * 4 + 1];
            bbox_delt[j][2] = bbox_delt_data[j * chw + i * 4 + 2];
            bbox_delt[j][3] = bbox_delt_data[j * chw + i * 4 + 3];
        }
        bbox_tranform_inv(rois_data, bbox_delt, num_roi,im.w - 1, im.w - 1);
        std::vector<abox> aboxes;
        for(int j = 0; j < num_roi; j++)
        {
            abox tmp;
            tmp.x1 = bbox_delt[j][0];
            tmp.y1 = bbox_delt[j][1];
            tmp.x2 = bbox_delt[j][2];
            tmp.y2 = bbox_delt[j][3];
            tmp.score = score[j * num_class + i];
            aboxes.push_back(tmp);
        }

        std::sort(aboxes.rbegin(), aboxes.rend());
        nms(aboxes, NMS_THRESH);
        for(int k = 0; k < ( int )aboxes.size();)
        {
            if(aboxes[k].score < CONF_THRESH)
                aboxes.erase(aboxes.begin() + k);
            else
                k++;
        }
        if(aboxes.size() > 0)
        {
            for(int b = 0; b < ( int )aboxes.size(); b++)
            {
                aboxes[b].class_idx = i;
                all_boxes.push_back(aboxes[b]);
            }
        }
    }
    for(int k = 0; k < num_roi; k++)
        free(bbox_delt[k]);
    free(bbox_delt);
    free(input_data);
    draw_detections(image_file, save_name, all_boxes);

    // test output data
    float* rois_data_ref = (float*)malloc(rois_size * sizeof(float));
    float* bbox_data_ref = (float*)malloc(bbox_size * sizeof(float));
    float* cls_data_ref = (float*)malloc(cls_size * sizeof(float));
    FILE *fp;  
    fp=fopen("./data/faster_rcnn_rois_out.bin","rb");
    if(fread(rois_data_ref, sizeof(float), rois_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return false;
    }
    fclose(fp);

    fp=fopen("./data/faster_rcnn_bbox_out.bin","rb");
    if(fread(bbox_data_ref, sizeof(float), bbox_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return false;
    }
    fclose(fp);

    fp=fopen("./data/faster_rcnn_cls_out.bin","rb");
    if(fread(cls_data_ref, sizeof(float), cls_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return false;
    }
    fclose(fp);
    
    if(mismatch_fp32(rois_data_ref, rois_data, rois_size, 0.002) != true)
        return -1;
    if(mismatch_fp32(bbox_data_ref, bbox_delt_data, bbox_size, 0.001) != true)
        return -1;
    if(mismatch_fp32(cls_data_ref, score, cls_size, 0.001) != true)
        return -1;

    release_graph_tensor(input_tensor1);
    release_graph_tensor(input_tensor2);
    release_graph_tensor(tensor);
    ret = postrun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Postrun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    destroy_graph(graph);
    release_tengine();
    return 0;
}
