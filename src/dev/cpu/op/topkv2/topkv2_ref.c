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

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include <math.h>
#include "topkv2_param.h"

struct topkv2_param_ref
{
    int k;
    int row_size;
    int num_rows;
};

static void swap_fp32(float* p, float* q)
{
    float buf;
    buf = *p;
    *p = *q;
    *q = buf;
    return;
}
static void swap_int(int* p, int* q)
{
    int buf;
    buf = *p;
    *p = *q;
    *q = buf;
    return;
}
static void quick_sort_fp32(float* a, int low, int high, int* indexv)
{
    int i = low;
    int j = high;
    float key = a[low];
    if (low >= high)    //如果low >= high说明排序结束了
    {
        return;
    }
    while (low < high)    //该while循环结束一次表示比较了一轮
    {
        while (low < high && key >= a[high])
        {
            --high;    //向前寻找
        }
        if (key < a[high])
        {
            swap_fp32(&a[low], &a[high]);
            swap_int(&indexv[low], &indexv[high]);
            // std::swap(indexv.at(low), indexv.at(high));
            ++low;
        }
        while (low < high && key <= a[low])
        {
            ++low;    //向后寻找
        }
        if (key > a[low])
        {
            swap_fp32(&a[low], &a[high]);
            swap_int(&indexv[low], &indexv[high]);
            // std::swap(indexv.at(low), indexv.at(high));
            --high;
        }
    }
    quick_sort_fp32(a, i, low - 1, indexv);    //用同样的方式对分出来的左边的部分进行同上的做法
    quick_sort_fp32(a, low + 1, j, indexv);    //用同样的方式对分出来的右边的部分进行同上的做法
}

static int ref_topkv2_fp32(float* in_data, float* out_data, int* out_index, struct topkv2_param_ref* param)
{
    int k = param->k;
    // fprintf(stderr, "K = %d  \n", k);
    // fprintf(stderr, "Num_rows = %d  \n", param->num_rows);
    // fprintf(stderr, "rows_size = %d  \n", param->row_size);

    int row_size = param->row_size;
    int num_rows = param->num_rows;
    // std::vector<int> index;
    int* index = ( int* )sys_malloc(row_size * sizeof(int));
    for (int i = 0; i < num_rows; ++i)
    {
        int start = i * row_size;
        //        fprintf(stderr, "start %d  \n",start );
        for (int j = 0; j < row_size; ++j)
            index[j] = j;

        //        fprintf(stderr, "size of the array - %d \n", (int)index.size());
        quick_sort_fp32(&in_data[start], 0, row_size - 1, index);
        //        fprintf(stderr, "*****************************\n");
        //        for(int a=0;a<row_size ;++a)
        //            fprintf(stderr, "Row-%d index %d  \n",i , index.at(a));
        //        fprintf(stderr, "*****************************\n");
        memcpy(&out_data[i * k], &in_data[start], k * sizeof(float));
        memcpy(&out_index[i * k], index, k * sizeof(float));
        sys_free(index);
        //        fprintf(stderr, "after clearsize of the array - %d \n", (int)index.size());
    }
    // for(int i = 0; i < num_rows * k; ++i)
    //{
    //    fprintf(stderr, "Value %f  \n", out_data[i]);
    //    fprintf(stderr, "Index %d  \n", out_index[i]);
    //}

    // fprintf(stderr, "size of the array - %d \n", ( int )index.size());
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct topkv2_param* _param = ( struct topkv2_param* )(ir_node->op.param_mem);
    struct ir_tensor* input_tensor;
    int out_nums = ir_node->output_num;
    struct topkv2_priv_info* topkv2_priv_info = ( struct topkv2_priv_info* )exec_node->ops_priv;
    // void** output_data=sys_malloc(2*sizeof(void));
    // for(int ii = 0; ii < 2; ++ii)
    // {
    //     output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[ii]);
    //     output_data[ii] = output_tensor->data;
    // }
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct ir_tensor* output_tensor_1 = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[1]);
    int dims_len = input_tensor->dim_num;
    // auto in_dims = input_tensor->GetShape().GetDim();
    int num_rows = 1;
    for (int i = 0; i < dims_len - 1; ++i)
    {
        num_rows *= input_tensor->dims[i];
    }
    struct topkv2_param_ref op_param;
    op_param.k = _param->k;
    op_param.row_size = input_tensor->dims[dims_len - 1];
    op_param.num_rows = num_rows;
    float* input = ( float* )input_tensor->data;
    int ret = ref_topkv2_fp32(input, ( float* )output_tensor->data, ( int* )output_tensor_1->data, &op_param);
    if (ret < 0)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_zeroslike_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_TOPKV2, &hcl_node_ops);
}

static int unreg_zeroslike_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_TOPKV2, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_zeroslike_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_zeroslike_hcl_ops);
