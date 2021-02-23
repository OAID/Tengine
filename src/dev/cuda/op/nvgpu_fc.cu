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
 * Author: hhchen@openailab.com
 */


#include "cuda_executor.hpp"

extern "C"
{
#include "tengine_op.h"
#include "fc_param.h"
}

void fc_gpu_kernel(cublasHandle_t& handle, struct ir_graph* ir_graph, struct ir_node* ir_node, dict_uint2voidx  gpu_addr_map)
{
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    const float* d_A =(const float*)gpu_addr_map[input_tensor->idx];
    const float* d_B =(const float*)gpu_addr_map[weight_tensor->idx];
    float* d_C =(float*)gpu_addr_map[output_tensor->idx];

    int hidden = input_tensor->dims[1];
    if (input_tensor->dim_num > 2)
        hidden = hidden * input_tensor->dims[2];
    if (input_tensor->dim_num > 3)
        hidden = hidden * input_tensor->dims[3];

    struct fc_param* fc_param = ( struct fc_param* )ir_node->op.param_mem;

    int A_ROW = input_tensor->dims[0];
    int A_COL = hidden;
    int B_ROW = hidden;
    int B_COL = fc_param->num_output;

    int need_trans = 0;
    int weight_out = weight_tensor->dims[0];
    if (weight_out ==fc_param->num_output)
        need_trans = 0;
    else
        need_trans = 1;

    float a = 1, b = 0;
    if (1 == need_trans)
    {
        cublasSgemm(
            handle,
            CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
            B_COL,          //矩阵B^T、C^T的行数
            A_ROW,          //矩阵A^T、C^T的列数
            B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,             //alpha的值
            d_B,            //左矩阵，为B^T
            B_COL,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A,            //右矩阵，为A^T
            A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,             //beta的值
            d_C,             //结果矩阵C
            B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );
    }
    else
    {
        cublasSgemm(
            handle,
            CUBLAS_OP_T,   //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,   //矩阵B的属性参数，转置
            B_COL,          //矩阵B^T、C^T的行数
            A_ROW,          //矩阵A^T、C^T的列数
            B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,             //alpha的值
            d_B,            //左矩阵，为B^T
            B_ROW,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A,            //右矩阵，为A^T
            A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,             //beta的值
            d_C,             //结果矩阵C
            B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );
    }
}

void CUDAEngine::AddFullyConnectionNode(struct ir_graph* ir_graph, struct ir_node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_FC.\n", ir_node->idx);
    cublasCreate(&this->cublas_handle);
    fc_gpu_kernel(this->cublas_handle, ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&fc_gpu_kernel, this->cublas_handle, ir_graph, ir_node, this->gpu_addr_map));
}
