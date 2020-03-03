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
 * Author: jjzeng@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

#include <unistd.h>
#include <iostream>
#include <string>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

struct ConcatInputParam
{
    int n;
    int c;
    int h;
    int w;
};

const char* gTypeName[] = {"Fp32", "Fp16", "int8", "uint8"};

float uint8_scale = 5.0f;
int uint8_zero = 2;

const char* gInputName[] = {"data1", "data2", "data3", "data4", "data5"};

int create_input_node(graph_t graph, const char* node_name, int n, int c, int h, int w, int layout, int data_type)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    // printf("create_input_node [%d]:[%d]:[%d]:[%d]\n",n,c,h,w);
    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    if(layout == 0)
    {
        int dims[4] = {n, c, h, w};
        set_tensor_shape(tensor, dims, 4);
    }
    else
    {
        int dims[4] = {n, h, w, c};
        set_tensor_shape(tensor, dims, 4);
    }

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_test_node(graph_t graph, const char* node_name, int axis, const char* inputs[], int input_counts)
{
    std::cout << "create_ggraph_node--" << axis << "\n";
    node_t test_node = create_graph_node(graph, node_name, "Concat");
    if(test_node == nullptr)
    {
        std::cout << "create_graph_node_failed : node_name=" << node_name << "\n";
        return -1;
    }

    int data_type = 0;
    int dim[4] = {0};
    int concat_dim = 0;
    for(int ii = 0; ii < input_counts; ++ii)
    {
        tensor_t input_tensor = get_graph_tensor(graph, inputs[ii]);
        data_type = get_tensor_data_type(input_tensor);

        get_tensor_shape(input_tensor, dim, 4);
        concat_dim += dim[axis];

        if(input_tensor == nullptr)
        {
            std::cout << "ERRNO: " << get_tengine_errno() << "\noutput";
            return -1;
        }
        // std::cout << "set_node_input_tensor\n";
        set_node_input_tensor(test_node, ii, input_tensor);
        // std::cout << "release_input_tensor\n";
        release_graph_tensor(input_tensor);
    }

    dim[axis] = concat_dim;
    std::cout << "output_dim: " << dim[0] << "-" << dim[1] << "-" << dim[2] << "-" << dim[3] << "\n";
    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    set_tensor_shape(output_tensor, dim, 4);
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    if(data_type == TENGINE_DT_UINT8)
    {
        set_tensor_quant_param(output_tensor, &uint8_scale, &uint8_zero, 1);
    }

    release_graph_tensor(output_tensor);

    set_node_attr_int(test_node, "axis", &axis);
    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph(const char* test_node_name, int axis, const std::vector<ConcatInputParam>& inputParam,
                          int layout, int data_type)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);
    printf("create_test_graph count:[%d]\n", ( int )inputParam.size());

    if(graph == nullptr)
    {
        std::cerr << "create failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_layout(graph, layout) < 0)
    {
        std::cerr << "set layout failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    for(std::size_t ii = 0; ii < inputParam.size(); ++ii)
    {
        if(create_input_node(graph, gInputName[ii], inputParam[ii].n, inputParam[ii].c, inputParam[ii].h,
                             inputParam[ii].w, layout, data_type) < 0)
        {
            std::cerr << "create input failed\n";
            return nullptr;
        }
    }

    // std::cout << "create_test_node\n";
    create_test_node(graph, test_node_name, axis, gInputName, ( int )inputParam.size());

    const char* outputs[] = {test_node_name};
    // std::cout << "set_graph_input_node\n";
    if(set_graph_input_node(graph, gInputName, ( int )inputParam.size()) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    // std::cout << "set_graph_output_node\n";
    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set outputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    return graph;
}

std::vector<void*> gMemList;
void* get_mem(int size)
{
    void* pBuf = malloc(size);
    gMemList.push_back(pBuf);
    return pBuf;
}

void release_all_mem()
{
    for(std::size_t ii = 0; ii < gMemList.size(); ++ii)
    {
        free(gMemList[ii]);
    }

    gMemList.clear();
}

void set_input_data(graph_t graph, int input_counts)
{
    for(int ii = 0; ii < input_counts; ++ii)
    {
        tensor_t input_tensor = get_graph_input_tensor(graph, ii, 0);

        int data_type = get_tensor_data_type(input_tensor);
        int buf_size = get_tensor_buffer_size(input_tensor);
        void* i_buf = get_mem(buf_size);

        int dims[4];

        int num = get_tensor_shape(input_tensor, dims, 4);
        int elem_num = 1;
        while(num != 0)
        {
            elem_num *= dims[num - 1];
            num--;
        }

        // std::cout << "set input data counts : " << elem_num << "\n";

        for(int i = 0; i < elem_num; i++)
        {
            int iVal = i + 1;
            if(i == elem_num - 1)
            {
                iVal = 127;
            }

            if(data_type == TENGINE_DT_FP32)
            {
                float* f = ( float* )i_buf;
                f[i] = iVal;
                // std::cout << f[i] << "   ";
            }
            else if(data_type == TENGINE_DT_FP16)
            {
                __fp16* f16 = ( __fp16* )i_buf;

#ifdef __ARM_ARCH
                f16[i] = iVal;
#else
                f16[i] = fp32_to_fp16(iVal);
#endif
                // std::cout << iVal << "   ";
            }
            else if(data_type == TENGINE_DT_UINT8)
            {
                uint8_t* i8 = ( uint8_t* )i_buf;
                i8[i] = iVal;
                // std::cout << (int)i8[i] << "   ";
            }
            else
            {
                int8_t* i8 = ( int8_t* )i_buf;
                i8[i] = iVal;
                // std::cout << (int)i8[i] << "   ";
            }

            /*if( i != 0 && i % 4 == 0 )
            {
                std::cout << "\n";
                }*/
        }
        // std::cout << "\n";

        set_tensor_buffer(input_tensor, i_buf, buf_size);
        if(data_type == TENGINE_DT_INT8 || data_type == TENGINE_DT_UINT8)
        {
            float scale = 1;
            int zero = 0;
            set_tensor_quant_param(input_tensor, &scale, &zero, 1);
        }

        release_graph_tensor(input_tensor);
    }
}

bool comp_fp32(float lv, float rv)
{
    return lv == rv;
}

bool comp_fp16(__fp16 lv, float rv)
{
    return fp16_to_fp32(lv) == rv;
}

bool comp_int8(int8_t lv, float rv, float scale)
{
    rv = (int8_t)(rv * scale);
    return lv == rv;
}

bool comp_uint8(uint8_t lv, float rv, float scale, int zero)
{
    uint8_t tmp = round(rv * scale + zero);
    return lv == tmp;
}

int comp_res(node_t out_1, node_t out_2, int data_type)
{
    tensor_t out1_tensor = get_node_output_tensor(out_1, 0);
    tensor_t out2_tensor = get_node_output_tensor(out_2, 0);

    int out1_size = get_tensor_buffer_size(out1_tensor);
    int out2_size = get_tensor_buffer_size(out2_tensor) / sizeof(float);

    int8_t* out1_buf = ( int8_t* )get_tensor_buffer(out1_tensor);
    int8_t* out2_buf = ( int8_t* )get_tensor_buffer(out2_tensor);

    if(data_type == TENGINE_DT_FP32)
    {
        int num = out1_size / sizeof(float);
        if(num != out2_size)
        {
            std::cout << "---------------------fp32 not match--------------------------\n";
        }

        float* pOut1 = ( float* )out1_buf;
        float* pOut2 = ( float* )out2_buf;
        for(int ii = 0; ii < num; ++ii)
        {
            if(!comp_fp32(pOut1[ii], pOut2[ii]))
            {
                std::cout << "---------------------fp32 not match--------------------------\n";
                return 1;
            }
        }
    }
    else if(data_type == TENGINE_DT_FP16)
    {
        int num = out1_size / sizeof(__fp16);
        if(num != out2_size)
        {
            std::cout << "---------------------fp16 not match--------------------------\n";
        }
        __fp16* pOut1 = ( __fp16* )out1_buf;
        float* pOut2 = ( float* )out2_buf;
        for(int ii = 0; ii < num; ++ii)
        {
            if(!comp_fp16(pOut1[ii], pOut2[ii]))
            {
                std::cout << "---------------------fp16 not match--------------------------\n";
                return 1;
            }
        }
    }
    else if(data_type == TENGINE_DT_INT8)
    {
        int num = out1_size;
        if(num != out2_size)
        {
            std::cout << "---------------------int8 not match--------------------------\n";
        }
        int8_t* pOut1 = out1_buf;
        float* pOut2 = ( float* )out2_buf;
        for(int ii = 0; ii < num; ++ii)
        {
            if(!comp_int8(pOut1[ii], pOut2[ii], 1))
            {
                std::cout << "---------------------int8 not match--------------------------\n";
                return 1;
            }
        }
    }
    else if(data_type == TENGINE_DT_UINT8)
    {
        int num = out1_size;
        if(num != out2_size)
        {
            std::cout << "---------------------uint8 not match--------------------------\n";
        }
        printf("uint8 count : [%d]\n", num);
        uint8_t* pOut1 = ( uint8_t* )out1_buf;
        float* pOut2 = ( float* )out2_buf;
        for(int ii = 0; ii < num; ++ii)
        {
            if(!comp_uint8(pOut1[ii], pOut2[ii], 1.0f / uint8_scale, uint8_zero))
            {
                std::cout << "---------------------uint8 not match--------------------------\n";
                return 1;
            }
        }
    }

    return 0;
}

int test_concat(const char* test_node_name, const std::vector<ConcatInputParam>& inputParam, int axis, int layout,
                int data_type)
{
    graph_t graph = create_test_graph(test_node_name, axis, inputParam, layout, data_type);
    graph_t graph1 = create_test_graph(test_node_name, axis, inputParam, layout, 0);
    if(graph == nullptr || graph1 == nullptr)
    {
        std::cout << "create graph failed!!!\n";
        return 1;
    }

    set_input_data(graph, ( int )inputParam.size());
    set_input_data(graph1, ( int )inputParam.size());

    setenv("OPS_REGISTRY", "reference", 1);
    setenv("OP_NAME", "Concat", 1);
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    unsetenv("OPS_REGISTRY");
    unsetenv("OP_NAME");

    if(prerun_graph(graph1) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    if(run_graph(graph, 1) < 0 || run_graph(graph1, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }

    node_t node_1 = get_graph_node(graph, test_node_name);
    node_t node_2 = get_graph_node(graph1, test_node_name);
    int ret = 0;
    ret = comp_res(node_1, node_2, data_type);

    release_graph_node(node_1);
    release_graph_node(node_2);

    postrun_graph(graph);

    destroy_graph(graph);

    release_all_mem();

    return ret;
}

#define __PUSH_PARAM__(_n, _c, _h, _w) \
    {                                  \
        ConcatInputParam param;        \
        param.n = _n;                  \
        param.c = _c;                  \
        param.h = _h;                  \
        param.w = _w;                  \
        vInput.push_back(param);       \
    }

class CVSResOut
{
    static const char* pDelimit;

public:
    CVSResOut(const char* fileName, const char* pTitles[], int count) : file(fileName, std::ios::out)
    {
        for(int ii = 0; ii < count; ++ii)
        {
            file << pTitles[ii] << pDelimit;
        }
        file << "\n";
    }
    void dump_res(const char* name, int axis, const char* des, const char* res)
    {
        file << name << pDelimit << axis << pDelimit << des << pDelimit << res << "\n";
    }

    void save()
    {
        file.close();
    }

private:
    std::fstream file;
};

const char* CVSResOut::pDelimit = " ";

void test(int layout, int axis, int data_type, const std::vector<int> dims, CVSResOut& rResFile)
{
    const char* test_node_name = "test_concat_op";
    std::vector<ConcatInputParam> vInput;
    std::vector<int> vDims = dims;
    printf("test 1-1 Concat axis : %d. dataType : %d\n", axis, data_type);
    __PUSH_PARAM__(dims[0], dims[1], dims[2], dims[3]);
    vDims[axis] += 1;
    __PUSH_PARAM__(dims[0], dims[1], dims[2], dims[3]);
    if(0 == test_concat(test_node_name, vInput, axis, layout, data_type))
    {
        printf("Axis[%d]-[%s]--------------------match-------------------\n", axis, gTypeName[data_type]);
        rResFile.dump_res(gTypeName[data_type], axis, "1-1", "match");
    }
    else
    {
        printf("Axis[%d]-[%s]--------------------not match-------------------\n", axis, gTypeName[data_type]);
        rResFile.dump_res(gTypeName[data_type], axis, "1-1", "not match");
    }

    vDims = dims;
    printf("test 1-1-1 Concat axis : %d. dataType : %d\n", axis, data_type);
    __PUSH_PARAM__(dims[0], dims[1], dims[2], dims[3]);
    vDims[axis] += 1;
    __PUSH_PARAM__(dims[0], dims[1], dims[2], dims[3]);
    vDims[axis] += 1;
    __PUSH_PARAM__(dims[0], dims[1], dims[2], dims[3]);
    if(0 == test_concat(test_node_name, vInput, axis, layout, data_type))
    {
        printf("Axis[%d]-[%s]--------------------match-------------------\n", axis, gTypeName[data_type]);
        rResFile.dump_res(gTypeName[data_type], axis, "1-1-1", "match");
    }
    else
    {
        printf("Axis[%d]-[%s]--------------------match-------------------\n", axis, gTypeName[data_type]);
        rResFile.dump_res(gTypeName[data_type], axis, "1-1-1", "not match");
    }
}

int main(int argc, char* argv[])
{
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NCHW;

    int res;
    while((res = getopt(argc, argv, "c:h:w:n:s:p:m:l:t:")) != -1)
    {
        switch(res)
        {
            case 't':
                data_type = strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }

    std::cout << "init_tengine\n";
    init_tengine();

    const char* pTitle[] = {"DatType", "Axis", "ConcatCounts", "Res"};
    CVSResOut ResOut("concat_ref_test.cvs", pTitle, 4);

    std::vector<int> vDims;
    vDims.push_back(2);
    vDims.push_back(3);
    vDims.push_back(2);
    vDims.push_back(2);

    for(int axis = 0; axis < 4; ++axis)
    {
        data_type = 0;
        test(layout, axis, data_type, vDims, ResOut);

        // fp16
        data_type++;
        test(layout, axis, data_type, vDims, ResOut);

        // int8
        data_type++;
        test(layout, axis, data_type, vDims, ResOut);

        // uint8
        data_type++;
        test(layout, axis, data_type, vDims, ResOut);
    }

    ResOut.save();
    release_tengine();

    return 0;
}
