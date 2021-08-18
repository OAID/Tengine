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

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @brief Convert tensor type to char array.
 *
 * @param [in]  tensor_type: The type of a tensor.
 *
 * @return  Tensor type char array.
 */
const char* get_tensor_type_string(int tensor_type);

/*!
 * @brief Convert tensor layout to char array.
 *
 * @param [in]  tensor_layout: The layout of a tensor.
 *
 * @return  Tensor layout char array.
 */
const char* get_tensor_layout_string(int tensor_layout);

/*!
 * @brief Convert model format to char array.
 *
 * @param [in]  model_format: The format of a model.
 *
 * @return  Model format char array.
 */
const char* get_model_format_string(int model_format);

/*!
 * @brief Convert operator name char array to enumeration value.
 *
 * @param [in]  name: The name of an operator.
 *
 * @return  Operator type enumeration value.
 */
int get_op_type_from_name(const char* name);

/*!
 * @brief Convert operator enumeration value to char array.
 *
 * @param [in]  op_type: The operator enumeration value.
 *
 * @return  Operator name char array.
 */
const char* get_op_name_from_type(int op_type);

/*!
 * @brief Get single element size of the tensor data type.
 *
 * @param [in]  data_type: The data type.
 *
 * @return  Size of single element.
 */
int get_tenser_element_size(int data_type);

/*!
 * @brief Convert tensor data type to char array.
 *
 * @param [in]  op_type: The tensor data type.
 *
 * @return  Tensor data type char array.
 */
const char* get_tensor_data_type_string(int data_type);

/*!
 * @brief Convert tensor data type single letter char array.
 *
 * @param [in]  op_type: The tensor data type.
 *
 * @return  Tensor data type char array of single letter.
 */
const char* data_type_typeinfo_name(int data_type);

void dump_float(const char* file_name, float* data, int number);

int get_mask_count(size_t mask);

int get_mask_index(size_t mask);

#ifdef __cplusplus
}
#endif
