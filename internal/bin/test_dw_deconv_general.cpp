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
 * Copyright (c) 2019, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <string>
#include <math.h>
#include "tengine_c_api.h"

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int k_size, int stride, int pad,
                     int in_c, int out_c, int group,int dilation)
{
    node_t conv_node = create_graph_node(graph, node_name, "Deconvolution");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(conv_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* weight */

    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    set_node_input_tensor(conv_node, 1, w_tensor);
    int w_dims[] = {in_c, out_c / group, k_size, k_size};

    set_tensor_shape(w_tensor, w_dims, 4);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    /* bias */
    std::string bias_name(node_name);
    bias_name += "/bias";

    node_t b_node = create_graph_node(graph, bias_name.c_str(), "Const");
    tensor_t b_tensor = create_graph_tensor(graph, bias_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
    int b_dims[] = {out_c};

    set_tensor_shape(b_tensor, b_dims, 1);

    set_node_input_tensor(conv_node, 2, b_tensor);
    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    /* attr */
    int pad1 = pad;
    set_node_attr_int(conv_node, "kernel_h", &k_size);
    set_node_attr_int(conv_node, "kernel_w", &k_size);
    set_node_attr_int(conv_node, "stride_h", &stride);
    set_node_attr_int(conv_node, "stride_w", &stride);
    set_node_attr_int(conv_node, "pad_h0", &pad);
    set_node_attr_int(conv_node, "pad_w0", &pad);
    set_node_attr_int(conv_node, "pad_h1", &pad1);
    set_node_attr_int(conv_node, "pad_w1", &pad1);
    set_node_attr_int(conv_node, "num_output", &out_c);
    set_node_attr_int(conv_node, "group", &group);
    set_node_attr_int(conv_node, "dilation_h", &dilation);
    set_node_attr_int(conv_node, "dilation_w", &dilation);

    release_graph_node(conv_node);

    return 0;
}
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

static inline void DumpFloat(const char *fname, float *data, int number)
{
    FILE *fp = fopen(fname, "w");

    for (int i = 0; i < number; i++) {
        if (i % 16*4 == 0) {
            fprintf(fp, "\n%d:",i);
        }
        fprintf(fp, " %.5f", data[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}

float maxerr(float *gt, float *pred, int size, int dump = 0)
{
    float maxError = 0.f;
    float tmp =0.f;
    for (int i = 0; i < size; i++) {
        tmp = (float)fabs(gt[i] - pred[i]);
        if (tmp > 0.01) {
            printf("==============================================\n");
            printf("mismatch at idx=%d, pred=%f, gt=%f\n", i, pred[i], gt[i]);
            printf("dump data to file [gt_data, pred_data]\n");
            printf("=============================================\n");
            DumpFloat("gt_data", gt, size);
            DumpFloat("pred_data", pred, size);
            return -1;
        }
        maxError = MAX(tmp, maxError);
    }
    // printf("maxerr %f\n",maxError);
    if (dump) {
        DumpFloat("gt_data", gt, size);
        DumpFloat("pred_data", pred, size);
    }
    return maxError;
}
graph_t create_conv_graph(int c, int h, int w, int ksize, int stride, int pad, int group,int dilation)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);
    //set_graph_layout(graph, TENGINE_LAYOUT_NHWC);

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* conv_name = "conv";

    if(create_input_node(graph, input_name, c, h, w) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_conv_node(graph, conv_name, input_name, ksize, stride, pad, c, group, group,dilation) < 0)
    {
        std::cerr << "create conv node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {conv_name};

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set outputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    return graph;
}

int main(int argc, char* argv[])
{
    int c=32, h=56, w=56, group=1;
    int dilation=1;
    int ksize=4;
    int stride=2;
    int pad=1;
    if(argc<9)
    {
        printf("./test [inc][inh][inw][group][dilation][ksize][stride][pad]\n [note]:inc should be divided by group!!!\n");
    }
    if(argc>=9)
    {
        c=atoi(argv[1]);
        h=atoi(argv[2]);
        w=atoi(argv[3]);
        group=atoi(argv[4]);
        dilation=atoi(argv[5]);
        ksize=atoi(argv[6]);
        stride=atoi(argv[7]);
        pad=atoi(argv[8]);
        if(h<ksize || w<ksize)
        {
            printf("h,w should >=ksize\n");
            return -1;
        }
        if (c%group!=0)
        {
            printf("c should be devided by group!\n");
            return -1;
        }
        if(pad>(ksize/2))
        {
            printf("pad should <=ksize/2\n");
            return -1;
        }
        if(stride>ksize)
        {
            printf("stride should <=ksize\n");
            return -1;
        }
    }

    init_tengine();
    graph_t graph = create_conv_graph(c, h, w, ksize,stride,pad,group,dilation);
    graph_t graph1 = create_conv_graph(c, h, w, ksize,stride,pad,group,dilation);

    if(graph == nullptr)
        return 1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t input_tensor1 = get_graph_input_tensor(graph1, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);
    float* i_buf1 = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        i_buf[i] =  0.01*(i % 10);
        i_buf1[i] =  0.01*(i % 10);
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);
    set_tensor_buffer(input_tensor1, i_buf1, buf_size);
    release_graph_tensor(input_tensor1);

    /* set weight */
    node_t conv_node = get_graph_node(graph, "conv");
    node_t conv_node1 = get_graph_node(graph1, "conv");

    tensor_t weight_tensor = get_node_input_tensor(conv_node, 1);
    tensor_t weight_tensor1 = get_node_input_tensor(conv_node1, 1);

    buf_size = get_tensor_buffer_size(weight_tensor);
    float* w_buf = ( float* )malloc(buf_size);
    float* w_buf1 = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        w_buf[i] = 0.01*(i%40);
        w_buf1[i] =0.01*(i%40);
    }

    set_tensor_buffer(weight_tensor, w_buf, buf_size);
    release_graph_tensor(weight_tensor);
    set_tensor_buffer(weight_tensor1, w_buf1, buf_size);
    release_graph_tensor(weight_tensor1);

    /* set bias */

    int input_num = get_node_input_number(conv_node);
    float* b_buf = nullptr;
    float* b_buf1 = nullptr;

    if(input_num > 2)
    {
        tensor_t bias_tensor = get_node_input_tensor(conv_node, 2);
        tensor_t bias_tensor1 = get_node_input_tensor(conv_node1, 2);

        buf_size = get_tensor_buffer_size(bias_tensor);
        b_buf = ( float* )malloc(buf_size);
        b_buf1 = ( float* )malloc(buf_size);

        for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
        {
            b_buf[i] = 1;
            b_buf1[i] = 1;
        }

        set_tensor_buffer(bias_tensor, b_buf, buf_size);
        release_graph_tensor(bias_tensor);
        set_tensor_buffer(bias_tensor1, b_buf1, buf_size);
        release_graph_tensor(bias_tensor1);
    }

    // prerun graph
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    setenv("OPS_REGISTRY", "reference", 1);
    setenv("OP_NAME", "Deconvolution", 1);
    prerun_graph(graph1);

    // dump_graph(graph);

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }
    run_graph(graph1,1);

    tensor_t output_tensor = get_node_output_tensor(conv_node, 0);
    tensor_t output_tensor1 = get_node_output_tensor(conv_node1, 0);

    float* buf = ( float* )get_tensor_buffer(output_tensor);
    float* buf1 = ( float* )get_tensor_buffer(output_tensor1);
    int out_size = get_tensor_buffer_size(output_tensor)/sizeof(float);
    float err=maxerr(buf1,buf,out_size);

    printf("c h w g = %d %d %d %d\t d k s p = %d %d %d %d\t maxerr = %f\n",c,h,w,group,dilation, ksize, stride, pad, err);
    if(err < 0.01)
        printf("test pass\n");
    else
        printf("test fail\n");
    release_graph_tensor(output_tensor);
    release_graph_node(conv_node);
    release_graph_tensor(output_tensor1);
    release_graph_node(conv_node1);

    postrun_graph(graph);
    postrun_graph(graph1);

    destroy_graph(graph);
    destroy_graph(graph1);

    free(i_buf);
    free(w_buf);
    free(i_buf1);
    free(w_buf1);

    if(b_buf)
    {
        free(b_buf);
        free(b_buf1);
    }

    release_tengine();
    return 0;
}
