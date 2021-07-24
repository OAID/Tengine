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

#ifndef __FC_KERNEL_FP16_ARM82_H_
#define __FC_KERNEL_FP16_ARM82_H_

#include "fc_kernel_arm.h"

int int8_fc_kernel_prerun(struct tensor* input_tensor,
                          struct tensor* filter_tensor,
                          struct tensor* output_tensor,
                          struct fc_priv_info* priv_info,
                          struct fc_param* param);

int int8_fc_kernel_run(struct tensor* input_tensor,
                       struct tensor* filter_tensor,
                       struct tensor* bias_tensor,
                       struct tensor* output_tensor,
                       struct fc_priv_info* priv_info,
                       struct fc_param* param,
                       int num_thread, int cpu_affinity);

#endif
