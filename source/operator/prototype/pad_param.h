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
 * Author: sqfu@openailab.com
 */

#pragma once

struct pad_param
{
    // mode : 0: CONSTANT; 1: REFLECT; 2: SYMMETRIC.
    int mode;
    int pad_0_h; // n
    int pad_0_w;
    int pad_1_h; // c
    int pad_1_w;
    int pad_2_h; // h
    int pad_2_w;
    int pad_3_h; // w
    int pad_3_w;
    float value; // pad value
};
