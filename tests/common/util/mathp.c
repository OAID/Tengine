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
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#include "mathp.h"

#include <stdlib.h>

int imin(int a, int b)
{
    return a <= b ? a : b;
}

int imax(int a, int b)
{
    return a >= b ? a : b;
}

int min_abs(int a, int b)
{
    return imin(abs(a), abs(b));
}

int max_abs(int a, int b)
{
    return imax(abs(a), abs(b));
}

static int solve_gcd(int large, int small)
{
    int val = large % small;
    return 0 == val ? small : gcd(small, val);
}

int gcd(int a, int b)
{
    if (0 == a || 0 == b)
        return 0;

    return solve_gcd(max_abs(a, b), min_abs(a, b));
}

int lcm(int a, int b)
{
    if (0 == a || 0 == b)
        return 0;

    return abs(a * b) / solve_gcd(max_abs(a, b), min_abs(a, b));
}

int align(int value, int step)
{
    const int mask = ~(abs(step) - 1);
    return (value + step) & mask;
}

void* align_address(void* address, int step)
{
    const size_t mask = ~(abs(step) - 1);
    return (void*)((size_t)address & mask);
}
