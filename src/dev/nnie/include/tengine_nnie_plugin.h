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
 * Author: cmeng@openailab.com
 */

#ifndef __TENGINE_NNIE_PLUGIN_H__
#define __TENGINE_NNIE_PLUGIN_H__

#include "tengine_c_api.h"
#ifdef __cplusplus
extern "C"
{
#endif
    /* For user to add user defined operator*/
    struct custom_op
    {
        const char *name; /* The name of the operator */
        void *param;      /* used for operator impl functions */
        int param_size;
        /*!
     * @brief Run the operator.
     *
     * @param [in] op: The point of custom defined operator.
     * @param [in] inputs[]: The custom defined input tensor.
     * @param [in] input_num: The number of the custom defined input tensor.
     * @param [in] outputs[]: The custom defined output tensor.
     * @param [in] output_num: The number of the custom defined output tensor.

     * @return 0: success, -1: fail.
     */
        int (*run)(struct custom_op *op, tensor_t inputs[], int input_num,
                   tensor_t outputs[], int output_num);
    };

    /*!
 * @brief Register customer operator.
 *
 * @param [in] op: The custom implemented  operater.
 *
 * @return 0: Success, -1: Fail.
 */
    int register_custom_op(struct custom_op *op);

    /*!
 * @brief Unregister customer.
 *
 * @param [in] op: The custom implemented  operater.
 *
 * @return 0: Success, -1: Fail.
 */
    int unregister_custom_op(struct custom_op *op);

#ifdef __cplusplus
}
#endif
#endif