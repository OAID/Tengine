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
 * Author: bhu@openailab.com
 */
static void swap_fp32(float* p, float* q)
{
    float buf;
    buf = *p;
    *p = *q;
    *q = buf;
    return;
}

static void quick_sort_fp32(float* a, int low, int high, std::vector<int>& indexv)
{
    int i = low;
    int j = high;
    float key = a[low];
    if(low >= high)    //如果low >= high说明排序结束了
    {
        return;
    }
    while(low < high)    //该while循环结束一次表示比较了一轮
    {
        while(low < high && key >= a[high])
        {
            --high;    //向前寻找
        }
        if(key < a[high])
        {
            swap_fp32(&a[low], &a[high]);
            std::swap(indexv.at(low), indexv.at(high));
            ++low;
        }
        while(low < high && key <= a[low])
        {
            ++low;    //向后寻找
        }
        if(key > a[low])
        {
            swap_fp32(&a[low], &a[high]);
            std::swap(indexv.at(low), indexv.at(high));
            --high;
        }
    }
    quick_sort_fp32(a, i, low - 1, indexv);    //用同样的方式对分出来的左边的部分进行同上的做法
    quick_sort_fp32(a, low + 1, j, indexv);    //用同样的方式对分出来的右边的部分进行同上的做法
}

static int ref_topkv2_fp32(float* in_data, float* out_data, int* out_index, struct topkv2_param* param)
{
    int k = param->k;
    //printf("K = %d  \n", k);
    //printf("Num_rows = %d  \n", param->num_rows);
    //printf("rows_size = %d  \n", param->row_size);

    int row_size = param->row_size;
    int num_rows = param->num_rows;
    std::vector<int> index;
    for(int i = 0; i < num_rows; ++i)
    {
        int start = i * row_size;
        //        printf("start %d  \n",start );
        for(int j = 0; j < row_size; ++j)
            index.push_back(j);

        //        printf("size of the array - %d \n", (int)index.size());
        quick_sort_fp32(&in_data[start], 0, row_size - 1, index);
        //        printf("*****************************\n");
        //        for(int a=0;a<row_size ;++a)
        //            printf("Row-%d index %d  \n",i , index.at(a));
        //        printf("*****************************\n");
        memcpy(&out_data[i * k], &in_data[start], k * sizeof(float));
        memcpy(&out_index[i * k], index.data(), k * sizeof(float));
        index.clear();
        //        printf("after clearsize of the array - %d \n", (int)index.size());
    }
    //for(int i = 0; i < num_rows * k; ++i)
    //{
    //    printf("Value %f  \n", out_data[i]);
    //    printf("Index %d  \n", out_index[i]);
    //}

    //printf("size of the array - %d \n", ( int )index.size());
    return 0;
}
