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
 * Author: hhchen@openailab.com
 */

#include "common.hpp"
#include "quant_tool_uint8.hpp"
#include "quant_save_graph.hpp"


QuantTool::QuantTool()
{
    // initial tengine
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
    }

    // system variable
    this->opt.num_thread = 4;
    this->opt.cluster = TENGINE_CLUSTER_ALL;
    this->opt.precision = TENGINE_MODE_FP32;
    this->opt.affinity = 0;

    this->num_thread = 4;

    // input variable
    this->img_chw[3] = {0.f};
    this->letterboxs[2] = {0.f};
    this->sw_RGB = 1;
    this->img_c = 3;
    this->img_h = 224;
    this->img_w = 224;
    this->mean[0] = 104.f;
    this->mean[1] = 117.f;
    this->mean[2] = 123.f;
    this->scale[0] = 1.f;
    this->scale[1] = 1.f;
    this->scale[2] = 1.f;
    this->center_crop = 0;
    this->letterbox_rows = 0;
    this->letterbox_cols = 0;
    this->focus = 0;
    this->inplace = true;
    this->algorithm_type = ALGORITHM_MIN_MAX;
}

QuantTool::~QuantTool()
{
    /* release tengine */
    release_tengine();
}

int QuantTool::activation_quant_tool(const char* model_file, const char* image_dir,
                                     int img_c, int img_h, int img_w, const float* mean, const float* scale,
                                     int num_thread, int sw_RGB, int center_crop, int letterbox_rows, int letterbox_cols, int focus)
{
    fprintf(stderr, "[Quant Tools Info]: Step 0, load FP32 tmfile.\n");

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    fprintf(stderr, "[Quant Tools Info]: Step 0, load FP32 tmfile done.\n");

    /* set the shape, data buffer of input_tensor of the graph */
    int img_size = img_h * img_w * img_c;
    int dims[] = {1, img_c, img_h, img_w};    // nchw
    float* input_data = ( float* )malloc(img_size * sizeof(float));

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

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* initial malloc the output tesnors date buffer of nodes in the graph, to disable the mem pool, before prerun */
    struct graph* graphn = ( struct graph* )graph;
    for (int i = 0; i < graphn->tensor_num; i++)
    {
        struct tensor* var_tensor = graphn->tensor_list[i];
        if (var_tensor->tensor_type == TENSOR_TYPE_VAR)
        {
            var_tensor->data = ( float* )malloc(sizeof(float));
        }
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, this->opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    fprintf(stderr, "[Quant Tools Info]: Step 0, load calibration image files.\n");

    /* really malloc the output tesnors date buffer of nodes in the graph */
    for (int i = 0; i < graphn->tensor_num; i++)
    {
        struct tensor* var_tensor = graphn->tensor_list[i];
        if (var_tensor->tensor_type == TENSOR_TYPE_VAR)
        {
            var_tensor->data = realloc(var_tensor->data, sizeof(float) * var_tensor->elem_num);
            memset(var_tensor->data, 0, sizeof(float) * var_tensor->elem_num);
        }
    }

    /* read image list */
    std::vector<std::string> imgs_list;
    if (image_dir != nullptr)
    {
        readFileList(image_dir, imgs_list);
    }

    uint32_t img_num = imgs_list.size();

    fprintf(stderr, "[Quant Tools Info]: Step 0, load calibration image files done, image num is %d.\n", img_num);

    /* init minmax */
    std::tr1::unordered_map<int, float> max_activation;
    std::tr1::unordered_map<int, float> min_activation;
    uint32_t act_tensor_num = 0;
    for (int i = 0; i < graphn->tensor_num; i++)
    {
        struct tensor* act_tensor = graphn->tensor_list[i];
        if (act_tensor->tensor_type == TENSOR_TYPE_VAR || act_tensor->tensor_type == TENSOR_TYPE_INPUT)
        {
            act_tensor_num++;
            max_activation[i] = -FLT_MAX;
            min_activation[i] = FLT_MAX;
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 1, find original calibration table.\n");

    /* first loop, find the min/max value of every activation tensor of the graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int nums = 0; nums < img_num; nums++)
    {
        fprintf(stderr, "\r[Quant Tools Info]: Step 1, images %.5d / %.5d", nums+1, img_num);
        // cv::Mat m = cv::imread(imgs_list[nums].c_str(), 1);
        get_input_data_cv(imgs_list[nums].c_str(), input_data, img_h, img_w, mean, scale, img_c, sw_RGB, center_crop, letterbox_rows, letterbox_cols, focus);

        /* run graph */
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);

        for (int i = 0; i < graphn->tensor_num; i++)
        {
            struct tensor* act_tensor = graphn->tensor_list[i];
            if (act_tensor->tensor_type == TENSOR_TYPE_VAR || act_tensor->tensor_type == TENSOR_TYPE_INPUT)
            {
                float* start_addr = ( float* )act_tensor->data;
                float* end_addr   = ( float* )act_tensor->data + act_tensor->elem_num;
                max_activation[i] = std::max(max_activation[i], *std::max_element(start_addr, end_addr));
                min_activation[i] = std::min(min_activation[i], *std::min_element(start_addr, end_addr));
            }
        }
    }

    /* save the calibration file with min-max algorithm */
    FILE* fp_minmax = fopen("table_minmax.scale", "wb");
    for (int i = 0; i < graphn->tensor_num; i++)
    {
        struct tensor* t = graphn->tensor_list[i];
        if (t->tensor_type == TENSOR_TYPE_VAR || t->tensor_type == TENSOR_TYPE_INPUT)
        {
            float act_scale;
            int act_zero_point;
            if (max_activation[i] < 0)
            {
                act_scale = (0 - min_activation[i]) / 255;
                act_zero_point = int(-min_activation[i] / act_scale);
            }
            else if (min_activation[i] > 0)
            {
                act_scale = (max_activation[i] - 0) / 255;
                act_zero_point = 0;                
            }
            else
            {
                act_scale = (max_activation[i] - min_activation[i]) / 255;
                act_zero_point = int(-min_activation[i] / act_scale);
            }
 
            if (act_scale == 0)
                act_zero_point = 0;

            /* the scale of softmax always is scale = 1 / 127.f */
            for (int j = 0; j < graphn->node_num; j++)
            {
                struct node* noden = graphn->node_list[j];
                struct tensor* tensor_tmp = get_ir_graph_tensor(graphn, noden->output_tensors[0]);

                if (!(tensor_tmp->tensor_type == TENSOR_TYPE_INPUT || tensor_tmp->tensor_type == TENSOR_TYPE_VAR))
                    continue;

                std::string tmp_op_name = get_op_name_from_type(noden->op.type);
                std::string cur_name = t->name;
                std::string tmp_name = tensor_tmp->name;

                if ((cur_name == tmp_name) && tmp_op_name == "Softmax")
                {
                    act_scale = 1 / 255.f;
                    act_zero_point = 0;
                    break;
                }
            }

            fprintf(fp_minmax, "%s %f %d\n", graphn->tensor_list[i]->name, act_scale, act_zero_point);
        }
    }
    fclose(fp_minmax);
    fprintf(stderr, "\r\n[Quant Tools Info]: Step 1, find original calibration table done, output ./table_minmax.scale\n");

    fprintf(stderr, "[Quant Tools Info]: Thread %d, image nums %d, total time %.2f ms, avg time %.2f ms\n", num_thread,
            img_num, total_time, total_time / img_num);

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);

    return 0;
}

const char* help_params = "[Quant Tools Info]: optional arguments:\n"
                          "\t-h    help            show this help message and exit\n"
                          "\t-m    input model     path to input float32 tmfile\n"
                          "\t-i    image dir       path to calibration images folder\n"
                          "\t-f    scale file      path to calibration scale file\n"
                          "\t-o    output model    path to output uint8 tmfile\n"
                          "\t-a    algorithm       the type of quant algorithm(0:min-max, 1:kl, default is 0)\n"
                          "\t-g    size            the size of input image(using the resize the original image,default is 3,224,224)\n"
                          "\t-w    mean            value of mean (mean value, default is 104.0,117.0,123.0)\n"
                          "\t-s    scale           value of normalize (scale value, default is 1.0,1.0,1.0)\n"
                          "\t-b    swapRB          flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)\n"
                          "\t-c    center crop     flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)\n"
                          "\t-y    letter box      flag which indicates that letter box process image is necessary(maybe using for YOLOv3/v4, 0:OFF, 1:ON, default is 0)\n"
                          "\t-k    focus           flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)\n"
                          "\t-t    num thread      count of processing threads(default is 1)\n";

const char* example_params = "[Quant Tools Info]: example arguments:\n"
                             "\t./quant_tool_uint8 -m ./mobilenet_fp32.tmfile -i ./dataset -o ./mobilenet_uint8.tmfile -g 3,224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017\n";

void show_usage()
{
    fprintf(stderr, "%s\n", help_params);
    fprintf(stderr, "%s\n", example_params);
}

int main(int argc, char* argv[])
{
    QuantTool quant_tool;

    int res;
    while ((res = getopt(argc, argv, "m:a:f:o:i:g:s:w:b:c:y:k:t:h")) != -1)
    {
        switch (res)
        {
            case 'm':
                quant_tool.model_file = optarg;
                break;
            case 'a':
                quant_tool.algorithm_type = atoi(optarg);
                break;
            case 'f':
                quant_tool.scale_file = optarg;
                break;
            case 'o':
                quant_tool.output_file = optarg;
                break;
            case 'i':
                quant_tool.image_dir = optarg;
                break;
            case 'g':
                split(quant_tool.img_chw, optarg, ",");
                quant_tool.img_c = ( int )quant_tool.img_chw[0];
                quant_tool.img_h = ( int )quant_tool.img_chw[1];
                quant_tool.img_w = ( int )quant_tool.img_chw[2];
                break;
            case 'w':
                split(quant_tool.mean, optarg, ",");
                break;
            case 's':
                split(quant_tool.scale, optarg, ",");
                break;
            case 'b':
                quant_tool.sw_RGB = atoi(optarg);
                break;
            case 'c':
                quant_tool.center_crop = atoi(optarg);
                break;
            case 'y':
                split(quant_tool.letterboxs, optarg, ",");
                quant_tool.letterbox_rows = ( int )quant_tool.letterboxs[0];
                quant_tool.letterbox_cols = ( int )quant_tool.letterboxs[1];
                break;
            case 'k':
                quant_tool.focus = atoi(optarg);
                break;                
            case 't':
                quant_tool.num_thread = atoi(optarg);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* version */
    fprintf(stderr, "\n---- Tengine Post Training Quantization Tool ---- \n");
    fprintf(stderr, "\nVersion     : v1.1, %s %s\n", __TIME__, __DATE__);
    fprintf(stderr, "Status      : uint8, per-layer, asymmetric\n");

    /* check input params */
    if (quant_tool.model_file.empty())
    {
        fprintf(stderr,"[Quant Tools Info]: The input file of Float32 tmfile file not specified!\n");
        show_usage();
        return -1;
    }

    if (quant_tool.image_dir.empty())
    {
        fprintf(stderr,"[Quant Tools Info]: The input dir of Calibration image not specified!\n");
        show_usage();
        return -1;
    }

    if (quant_tool.output_file.empty())
    {
        fprintf(stderr,"[Quant Tools Info]: The output file of Int8 tmfile not specified!\n");
        show_usage();
        return -1;
    }

    /* debug info : input params */
    fprintf(stderr, "Input model : %s\n", quant_tool.model_file.c_str());
    fprintf(stderr, "Output model: %s\n", quant_tool.output_file.c_str());
    fprintf(stderr, "Calib images: %s\n", quant_tool.image_dir.c_str());
    fprintf(stderr, "Scale file  : %s\n", quant_tool.scale_file.empty()?"NULL":quant_tool.scale_file.c_str());
    fprintf(stderr, "Algorithm   : %s\n", quant_tool.algorithm_type?"KL":"MIN MAX");
    fprintf(stderr, "Dims        : %d %d %d\n", quant_tool.img_c, quant_tool.img_h, quant_tool.img_w);
    fprintf(stderr, "Mean        : %.3f %.3f %.3f\n", quant_tool.mean[0], quant_tool.mean[1], quant_tool.mean[2]);
    fprintf(stderr, "Scale       : %.3f %.3f %.3f\n", quant_tool.scale[0], quant_tool.scale[1], quant_tool.scale[2]);
    fprintf(stderr, "BGR2RGB     : %s\n", quant_tool.sw_RGB?"ON":"OFF");
    fprintf(stderr, "Center crop : %s\n", quant_tool.center_crop?"ON":"OFF");
    fprintf(stderr, "Letter box  : %.0f %.0f\n", quant_tool.letterboxs[0], quant_tool.letterboxs[1]);
    fprintf(stderr, "YOLOv5 focus: %s\n", quant_tool.focus?"ON":"OFF");
    fprintf(stderr, "Thread num  : %d\n\n", quant_tool.num_thread);

    /* quantize activation */
    quant_tool.activation_quant_tool(quant_tool.model_file.c_str(), quant_tool.image_dir.c_str(), quant_tool.img_c, quant_tool.img_h, quant_tool.img_w, quant_tool.mean, quant_tool.scale,
                                     quant_tool.num_thread, quant_tool.sw_RGB, quant_tool.center_crop, quant_tool.letterbox_rows, quant_tool.letterbox_cols, quant_tool.focus);

    /* using 3rd calibration table file */
    if (quant_tool.scale_file.empty())
    {
        /* select algorithm */
        if (quant_tool.algorithm_type == ALGORITHM_MIN_MAX)
            quant_tool.scale_file = "table_minmax.scale";
        else if  (quant_tool.algorithm_type == ALGORITHM_KL)
            quant_tool.scale_file = "table_kl.scale";
        else
        {
            fprintf(stderr,"[Quant Tools Info]: algorithm not specified, using default type MIN MAX\n");
            quant_tool.scale_file = "table_kl.scale";
        }
    }

    /* quantize weight/bias and save into uint8 tmfile */
    fprintf(stderr,"[Quant Tools Info]: Calibration file is using %s\n", quant_tool.scale_file.c_str());
    save_graph_u8_perlayer(quant_tool.model_file.c_str(), quant_tool.scale_file.c_str(), quant_tool.output_file, quant_tool.inplace, quant_tool.num_thread, false);

    fprintf(stderr, "\n---- Tengine Int8 tmfile create success, best wish for your INT8 inference has a low accuracy loss...\\(^0^)/ ----\n");

    return 0;
}
