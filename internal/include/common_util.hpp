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
#ifndef __COMMON_UTIL_HPP__
#define __COMMON_UTIL_HPP__

#include <algorithm>
#include <vector>
#include <utility>
#include <stdio.h>

namespace TEngine {

static inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for(size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for(int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

static inline void DumpFloat(const char* fname, float* data, int number)
{
    FILE* fp = fopen(fname, "w");

    for(int i = 0; i < number; i++)
    {
        if(i % 16 == 0)
        {
            fprintf(fp, "\n%d:", i);
        }
        fprintf(fp, " %.5f", data[i]);
    }

    fprintf(fp, "\n");

    fclose(fp);
}

}    // namespace TEngine

#endif
