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

#include "quant_tool.hpp"


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
#if 0
    // basic messge
    this->img_size = 0;
    this->cosin_max = -9999.999f;
    this->scale_acc = 1.f;

    // ir graph variable
    this->fp32_out.clear();
    this->fake_quant_out.clear();
    this->input_datas_fp32.clear();
    this->input_datas_fake_quant.clear();
    this->out_imgs_fp32.clear();
    this->out_imgs_fake_quant.clear();

    this->graphn_fp32 = nullptr;
    this->graphn_fake_quant = nullptr;
    this->exec_graph_fp32 = nullptr;
    this->exec_graph_fake_quant = nullptr;
    this->exec_node_num = 0;

    // temp variable
    this->node_fp32 = nullptr;
    this->node_fake_quant = nullptr;
    this->node_ops_fp32 = nullptr;
    this->node_ops_fake_quant = nullptr;

    this->input_tensor_fp32 = nullptr;
    this->input_tensor_fake_quant = nullptr;
    this->weight_tensor_fp32 = nullptr;
    this->weight_tensor_fake_quant = nullptr;
    this->bias_tensor_fp32 = nullptr;
    this->bias_tensor_fake_quant = nullptr;
    this->output_tensor_fp32 = nullptr;
    this->output_tensor_fake_quant = nullptr;

    this->weight_data_fp32 = nullptr;
    this->weight_data_fake_quant = nullptr;
    this->weight_size = 0;
    this->interleave_buffer_fp32 = nullptr;
    this->interleave_buffer_fake_quant = nullptr;
    this->interleave_size_fake = 0;
    this->bias_data_fp32 = nullptr;
    this->bias_data_fake_quant = nullptr;
    this->bias_size = 0;

    this->conv_priv_info_fp32 = nullptr;
    this->conv_priv_info_fake_quant = nullptr;
    this->conv_param_fp32 = nullptr;
    this->conv_param_fake_quant = nullptr;
#endif
}

QuantTool::~QuantTool()
{
    /* release tengine */
    release_tengine();
}
#if 0
int QuantTool::init()
{
    // ir graph variable
    this->fp32_out.clear();
    this->fake_quant_out.clear();

    /* load fp32 graph and fake quant graph */
    this->graphn_fp32 = ( struct graph* )create_graph(nullptr, "tengine", this->model_file.c_str());
    this->graphn_fake_quant = ( struct graph* )create_graph(nullptr, "tengine", this->model_file.c_str());

    if (this->graphn_fp32 == nullptr || this->graphn_fake_quant == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    /* load activation scale to ir_tensor */
    this->load_activation_scale(this->graphn_fp32, this->scale_file.c_str(), this->inplace);
    this->load_activation_scale(this->graphn_fake_quant, this->scale_file.c_str(), this->inplace);

    /* get graph input tensor */
    this->graph_input_tensor_fp32 = ( struct tensor* )get_graph_input_tensor(( void* )this->graphn_fp32, 0, 0);
    this->graph_input_tensor_fake_quant =
        ( struct tensor* )get_graph_input_tensor(( void* )this->graphn_fake_quant, 0, 0);
    if (this->graph_input_tensor_fp32 == nullptr || this->graph_input_tensor_fake_quant == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    /* generate images list */
    std::vector<std::string> imgs_list;
    if (!this->image_dir.empty())
        readFileList(this->image_dir, imgs_list);
    else
        imgs_list.push_back(image_file);
    uint32_t img_num = imgs_list.size();
//    printf("### img_num %d\n", img_num);
    if (img_num < this->max_search_img_num)
        this->max_search_img_num = img_num;

    /* set the shape, data buffer of input_tensor of the graph */
    this->img_size = this->img_h * this->img_w * this->img_c;
    int dims[] = {1, img_c, img_h, img_w};    // nchw
    float* input_data_fp32 = ( float* )malloc(this->img_size * sizeof(float));
    float* input_data_fake_quant = ( float* )malloc(this->img_size * sizeof(float));

    /* prepare process input data, set the data mem to input tensor */
    float scale_graph_input = this->graph_input_tensor_fake_quant->scale;
    int zero_point_graph_input = this->graph_input_tensor_fake_quant->zero_point;
//    fprintf(stderr, "scale zp %f %d\n", scale_graph_input, zero_point_graph_input);

    this->input_datas_fp32.resize(this->max_search_img_num);
    this->input_datas_fake_quant.resize(this->max_search_img_num);
    cv::Mat m;
    for (int i = 0; i < this->max_search_img_num; i++)
    {
        this->input_datas_fp32[i].resize(this->img_size);
        this->input_datas_fake_quant[i].resize(this->img_size);

        get_input_data_cv(imgs_list[i].c_str(), this->input_datas_fp32[i].data(), this->img_h, this->img_w, this->mean, this->scale,
                          this->img_c, this->sw_RGB, this->center_crop, this->letterbox_rows, this->letterbox_cols, this->focus);

        this->input_datas_fake_quant[i] = this->input_datas_fp32[i];
        this->activation_requant(this->input_datas_fake_quant[i].data(), this->img_size, 8, 1, scale_graph_input,
                                 zero_point_graph_input);
    }

    /* set graph input shape */
    int ret_fp32 = set_tensor_shape(this->graph_input_tensor_fp32, dims, 4);
    int ret_fake_quant = set_tensor_shape(this->graph_input_tensor_fake_quant, dims, 4);
    if (ret_fp32 < 0 || ret_fake_quant < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    /* set graph input buffer */
    ret_fp32 = set_tensor_buffer(this->graph_input_tensor_fp32, input_data_fp32, this->img_size * 4);
    ret_fake_quant = set_tensor_buffer(this->graph_input_tensor_fake_quant, input_data_fake_quant, this->img_size * 4);
    if (ret_fp32 < 0 || ret_fake_quant < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    ret_fp32 = this->prerun_for_get_ir_tensor(( void* )this->graphn_fp32, this->opt);
    ret_fake_quant = this->prerun_for_get_ir_tensor(( void* )this->graphn_fake_quant, this->opt);
    if (ret_fp32 < 0 || ret_fake_quant < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* get exec graph */
    this->exec_graph_fp32 = this->get_exec_graph(this->graphn_fp32);
    this->exec_graph_fake_quant = this->get_exec_graph(this->graphn_fake_quant);
    this->exec_node_num = get_vector_num(this->exec_graph_fp32->exec_node_list);

    /* ir idx <<<->>> exec idx */
    for (int i = 0; i < this->exec_node_num; i++)
    {
        this->node_fp32 = ( struct exec_node* )get_vector_data(this->exec_graph_fp32->exec_node_list, i);
        this->node_fake_quant = ( struct exec_node* )get_vector_data(this->exec_graph_fake_quant->exec_node_list, i);

        int out_t = node_fp32->ir_node->output_tensors[0];
        this->ir_exec[graphn_fp32->tensor_list[out_t]->producer] = i;    // ir idx --> exec idx
        this->exec_ir[i] = graphn_fp32->tensor_list[out_t]->producer;    // exec idx --> ir idx
//        printf(" %d : %d\n", graphn_fp32->tensor_list[out_t]->producer, i);
    }

    /* check for free node*/
    this->check_for_free();

    return 0;
}


#endif

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
