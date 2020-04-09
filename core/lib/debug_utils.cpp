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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <iostream>
#include <cstdio>
#include <cmath>

#include "debug_utils.hpp"

namespace TEngine {

void DumpFloat(const char* fname, float* data, int number)
{
    FILE* fp = fopen(fname, "w");

    for(int i = 0; i < number; i++)
    {
        if(i % 16 == 0)
        {
            fprintf(fp, "\n%d:", i);
        }
        fprintf(fp, " %.2f", data[i]);
    }

    fprintf(fp, "\n");

    fclose(fp);
}

std::string ReplaceChar(const std::string& src, char from, char to)
{
    std::string ret;

    int size = src.size();

    for(int i = 0; i < size; i++)
    {
        char c = src[i];

        if(c == from)
            ret.push_back(to);
        else
            ret.push_back(c);
    }

    return ret;
}

// calculate max_error between pred_data and ground_truth_data
void CalcMaxError(float* pred, float* gt, int size)
{
    float maxError = 0.f;
    for(int i = 0; i < size; i++)
    {
        maxError = std::max(( float )fabs(gt[i] - pred[i]), maxError);
    }
    std::cout << "====================================\n ";
    std::cout << "maxError is " << maxError << std::endl;
    std::cout << "====================================\n ";
}
bool CompareFloatTensor(float* a, float* b, std::vector<int>& shape_dim, std::vector<int>& mismatch_dim)
{
    int total_size = 1;

    for(unsigned int i = 0; i < shape_dim.size(); i++)
    {
        total_size *= shape_dim[i];
    }

    int n = 0;

    while(n < total_size)
    {
        if(a[n] == b[n])
            n++;
        else
        {
            std::cout << "a is: " << a[n - 1] << " " << a[n] << " " << a[n + 1] << "\n";
            std::cout << "b is: " << b[n - 1] << " " << b[n] << " " << b[n + 1] << "\n";
            std::cout << "off: " << a[n] - b[n] << "\n";
            break;
        }
    }

    if(n == total_size)
        return true;

    int dim_size = shape_dim.size();

    mismatch_dim.resize(dim_size);

    int dim_len = 1;

    for(int i = shape_dim.size() - 1; i >= 0; i--)
    {
        mismatch_dim[i] = (n / dim_len) % shape_dim[i];
        dim_len *= shape_dim[i];
    }

    return false;
}

}    // namespace TEngine
