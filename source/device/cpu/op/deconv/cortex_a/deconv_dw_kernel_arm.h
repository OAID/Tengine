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
 * Author: haoluo@openailab.com
 */
#ifndef __DECONV_DW_KERNEL_ARM_H_
#define __DECONV_DW_KERNEL_ARM_H_

#include "deconv_param.h"

#include "graph/tensor.h"

int deconv_dw_run(struct tensor* input_tensor,
                  struct tensor* filter_tensor,
                  struct tensor* bias_tensor,
                  struct tensor* output_tensor,
                  struct deconv_param* param,
                  int num_thread,
                  int cpu_affinity);

#endif
